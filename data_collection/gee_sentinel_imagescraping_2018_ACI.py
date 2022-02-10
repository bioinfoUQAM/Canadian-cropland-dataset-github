# Creation of a Sentinel-2 Canadian Dataset

"""
# Note: before running this code, create a new conda env named : gee

# conda install -c anaconda pandas
# conda install -c conda-forge pysimplegui
# conda install openpyxl
# conda install -c conda-forge earthengine-api

# Then, authenticate yourself to Google Eeath Engine
# earthengine authenticate
# Inspiration: https://climada-python.readthedocs.io/en/stable/tutorial/climada_util_earth_engine.html

# Amanda Boatswain Jacques / Etienne Lord

# Since 20 March 2021
# Last updated : 2021-04-22 5:55 PM
"""

# Library imports
from datetime import datetime
import ee
import os
import pandas as pd
import PySimpleGUI as sg
import sys
import time
import webbrowser

# Initialize Earth Engine
ee.Initialize()


""" CONSTANTS"""
ACI_dict = {row[0] : row[1] for key, row in pd.read_csv("aci_crop_classifications_iac_classifications_des_cultures.csv").iterrows()}

year_interval = ["2018-01-01","2018-12-31"]
year = "2018"

""" Load data from .csv """
dataset = pd.read_csv("points_ALL_categories_2018.csv")



""" Helper Functions  """


# Create a rectangle region using EPSG:3978, radius in meters, point is Lat,Lng
def makeRectangle(point, xRadius=320, yRadius=320):
    proj = "EPSG:3978";
    pointLatLon = ee.Geometry.Point(point);
    pointMeters = pointLatLon.transform(proj, 0.001);
    coords = pointMeters.coordinates();
    minX = ee.Number(coords.get(0)).subtract(xRadius);
    minY = ee.Number(coords.get(1)).subtract(yRadius);
    maxX = ee.Number(coords.get(0)).add(xRadius);
    maxY = ee.Number(coords.get(1)).add(yRadius);
    rect = ee.Geometry.Rectangle([minX, minY, maxX, maxY], proj, False);
    return rect;


# Create a median image over a time range
def obtain_image_sentinel2(time_range, area):

    def maskclouds(image):
        band_qa = image.select("QA60")
        cloud_mask = ee.Number(2).pow(10).int()
        cirrus_mask = ee.Number(2).pow(11).int()

        # Remove all the pixels that are either cloud or shadows
        mask = band_qa.bitwiseAnd(cloud_mask).eq(0) and(
            band_qa.bitwiseAnd(cirrus_mask).eq(0))

        # (Would be wise to rewrite this as a threshold, and not just exclude
        # all pixels of a given type....)
        return image.updateMask(mask).divide(10000)

    # Create the median image for that particular season and point
    collection = ee.ImageCollection("COPERNICUS/S2")
    sentinel_filtered = (ee.ImageCollection(collection).
                         filterBounds(area).
                         filterDate(time_range[0], time_range[1]).
                         filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5)).map(maskclouds))

    sentinel_median = sentinel_filtered.median()
    return sentinel_median

def extract_ACI_id(year, point):
    ACI = ee.ImageCollection("AAFC/ACI")
    point = ee.Geometry.Point(point)
    ACI_img = ACI.select("landcover").filter(ee.Filter.date(year_interval[0], year_interval[1])).first()
    crop_id = ACI_img.reduceRegion(ee.Reducer.first(), point, 10, bestEffort = True).get("landcover").getInfo()
    #print("Crop Inventory ID: ", crop_id)

    try:
        crop_type = ACI_dict[crop_id].upper()

    except KeyError as e:
        # If there is no ID associated with that pixel, return "Unknown"
        crop_type = "Unknown".upper()
        crop_id = "None"

    return crop_id, crop_type

# Specify the correct encoding and region
def get_url(name, image, scale, region):
    path = image.getDownloadURL({
        "name":(name),
        "scale": scale,
        "region":(region),
        "crs":"EPSG:3978"
        })
    return path


""" DEFINE THE VEGETATION INDICES TO BE CALCULATED"""

def addGNDVI(image):
    gndvi = image.expression('((RE3 - GREEN)/(RE3 + GREEN))',
                                 {'RE3': image.select('B7'),
                                  'GREEN': image.select('B3')}).rename('GNDVI')
    return image.addBands(gndvi)

def addNDVI45(image):
    ndvi45 = image.expression('((RE1 - RED)/(RE1 + RED))',
                                 {'RE1': image.select('B5'),
                                  'RED': image.select('B4')}).rename('NDVI45')
    return image.addBands(ndvi45)

def addNDVI(image):
    ndvi = image.expression('((RE3 - RED)/(RE3 + RED))',
                                 {'RE3': image.select('B7'),
                                  'RED': image.select('B4')}).rename('NDVI')
    return image.addBands(ndvi)

def addOSAVI(image):
    osavi = image.expression('((1 + 0.16)*(NIR - RED)/(NIR + RED + 0.61))',
                                 {'NIR': image.select('B8'),
                                  'RED': image.select('B4')}).rename('OSAVI')
    return image.addBands(osavi)

def addPSRI(image):
    psri = image.expression('((RED-BLUE)/(RE2))',
                                 {'RE2': image.select('B6'),
                                  'BLUE': image.select('B2'),
                                  'RED': image.select('B4')}).rename('PSRI')
    return image.addBands(psri)



""" Main Loop """
# create the GUI layout
layout = [[sg.Text("Start downloading samples?")], [sg.Button("Yes")], [sg.Button("Quit")]]
data= []
invalid_files = []
batch_size = 24920

def main():

    iterations = 0

    while True:
        event, values = sg.Window(margins=(60, 20), title = 'Choose an option', layout= layout).read(close=True)

        if event == "Yes":
            print("[INFO] Downloading images in batches of {}".format(batch_size))

            # loop through the rows of the dataframe
            for i, row in dataset.iterrows():
                # extract all the information from the dataframe

                start_date = row["Month Start Date"]
                end_date = row["Month End Date"]
                #filename = row["File Name"].split(".zip")[0]
                lon = row["Longitude"]
                lat = row["Latitude"]
                #available = True
                region = row["Region"]
                #aci_ID = row["ACI Crop ID"]
                point_ID = row["Point ID"]
                prov_code = row["Province Code"]
                #crop_type = row["Crop Type"]
                point = [lon, lat]

                aci_id, crop_type =  extract_ACI_id(year_interval, point)


                filename = "POINT_" + str(point_ID) + "_" + start_date.split("-")[0] + start_date.split("-")[1] + "_" + prov_code + "_" + crop_type

                # Debugging
                #print("i: ", i)
                #print("row: ", row)
                #print("lat {}, lon {}".format(lat, lon))
                #print("Available: ", available)

                try:
                    # Create a rectangle around the coordinate and create the image in Earth Engine
                    area_sentinel = makeRectangle(point)
                    time_range_season = [start_date,  end_date]
                    composite_sentinel = obtain_image_sentinel2(time_range_season, area_sentinel)

                    # Add new bands for vegetation indices
                    composite_sentinel = addGNDVI(composite_sentinel)
                    composite_sentinel = addNDVI(composite_sentinel)
                    composite_sentinel = addNDVI45(composite_sentinel)
                    composite_sentinel = addOSAVI(composite_sentinel)
                    composite_sentinel = addPSRI(composite_sentinel)


                    # Check if the file already exists, if doesn't, get the url and download it
                    if (not os.path.exists(filename)):
                        url = get_url(filename, composite_sentinel, 10, area_sentinel)

                        #Debugging
                        #print("filename: ", filename)
                        #print("url: ", url)

                        #print("Available: ", available)
                        webbrowser.open_new_tab(url)
                        available = True
                        time.sleep(1)
                        print("[INFO] Downloading file {}".format(filename))
                        pass

                except Exception as e:
                    # skip the locations and time combos that do not have all the available bands
                    print("[Error] " + filename + " not downloaded because of " + str(e))
                    invalid_files.append(filename)
                    available = False
                    pass

                new_row = [point_ID, region, prov_code, crop_type.lower(), year, start_date, end_date, aci_id, lon, lat, filename, available]
                #print("New Row: ", new_row)
                data.append(new_row)



                iterations +=1
                print("Iteration: ", iterations)
                if  iterations % batch_size == 0:

                    print("[Info] Waiting for samples to download.")
                    # Pause the program for a few seconds
                    time.sleep(2)

                    confirm = sg.PopupYesNo("Continue with next batch?")
                    if confirm == 'Yes':
                        continue

                    else:
                        return data, False

                        break

        else:
            sg.popup("Quitting Program")
            break

    return data

final_dataframe = main()


final_dataframe = pd.DataFrame(final_dataframe, columns = ["Point ID", "Region", "Province Code",
                                                        "Crop Type", "Year", "Month Start Date", "Month End Date", "ACI Crop ID",
                                                        "Longitude", "Latitude", "File Name", "Available"])

final_dataframe.to_csv("points_ALL_categories_2018_ACI_2.csv")
