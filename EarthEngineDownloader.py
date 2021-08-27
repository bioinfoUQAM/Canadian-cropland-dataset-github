# Creation of a Sentinel-2 Canadian Dataset

"""
# Note: before running this code, create a new conda env named : gee

# conda install -c anaconda pandas
# conda install -c conda-forge pysimplegui
# conda install openpyxl
# conda install -c conda-forge earthengine-api

# Then, authenticate yourself to Google Eeath Engine (GEE)
# earthengine authenticate
# Inspiration: https://climada-python.readthedocs.io/en/stable/tutorial/climada_util_earth_engine.html

# Amanda Boatswain Jacques / Etienne Lord

# Since 20 March 2021
# Last updated : 2021-06-16 6:00 PM
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


class EarthEngineDownloader:
    
    """ This is class to assist the downloading of images directly from Earth 
    Engine. It takes as input a .csv file with geographical coordinates of fields. 
    It also accepts a timestamp as input, which can be used to generate images at 
    weekly intervals. """
    
    def __init__(self):
        pass    
    
    def download_image(self, point_ID, lon, lat, time_interval):
        # use this function when downloading an image for a single point 
        """
        point_ID: string 
        lon, lat: float
        time_interval: 2-item date list of the format ["YYYY-MM-DD", "YYYY-MM-DD"], 
        where the first date precedes the second date. 
        """
        
        filename = point_ID + "_" + time_interval[0].split("-")[0] + time_interval[0].split("-")[1]
        #print("filename: ", filename)
        
        try:
             # Create a rectangle around the coordinate and create the image in Earth Engine
             area_sentinel = make_rectangle([lon, lat])
             composite_sentinel = obtain_image_sentinel2(time_interval, area_sentinel)
             
             # Add new bands for vegetation indices
             composite_sentinel = addGNDVI(composite_sentinel)
             composite_sentinel = addNDVI45(composite_sentinel)
             composite_sentinel = addOSAVI(composite_sentinel)
             composite_sentinel = addPSRI(composite_sentinel)
             #print("filename: ", filename+".zip")
             
             url = get_url(filename, composite_sentinel, 10, area_sentinel)
             
             webbrowser.open_new_tab(url)
             print("[INFO] Downloading file as {}".format(filename))
            

        except Exception as e:
                # skip the locations and time combos that do not have all the available bands
                print("[Error] " + filename + " not downloaded because of " + str(e))
     
    def download_images_from_file(self, file_path, time_interval, name_fields, download_path=None):
        """
        file_path: .csv file of geographic coordinates. At the minimum, the file 
        needs to include a Longitude, Latitude, and "ID" field.
        time_interval: 2-item date list of the format ["YYYY-MM-DD", "YYYY-MM-DD"]
        name_fields: list of fields in the .csv that need to be included in the file name
        download_path: path to the downloads folder 
         """
        
        # read the data from the text file
        self.data = pd.read_csv(file_path)

        # loop through the rows of the dataframe and extract the required information
        for i, row in self.data.iterrows():
            point_ID = "_".join(row[name_fields])
            lon = row["Longitude"]
            lat = row["Latitude"]
            
            """
            Add some code to check if the file exists in the current download path
            """            
            
            self.download_image(point_ID, lon, lat, time_interval)
            # add a short delay before download another file 
            time.sleep(2)



""" HELPER FUNCTIONS """

""" NOT FULLY FUNCTIONAL YET
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
"""

# Create a rectangle region using EPSG:3978, radius in meters, point is Lng, Lat
def make_rectangle(point, xRadius=320, yRadius=320):
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

# Create a median image over a specified time range
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


# Specify the correct encoding and region, get download path of the image 
def get_url(name, image, scale, rect):
    path = image.getDownloadURL({
        "name":(name),
        "scale": scale,
        "region":(rect),
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
