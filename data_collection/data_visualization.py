# Creation of a Sentinel-2 Canadian Dataset

"""
# Note: before running this code, create a new conda env named : gee

conda install -c anaconda pandas
conda install -c anaconda pillow
conda install -c conda-forge matplotlib


# Then, authenticate yourself to Google Eeath Engine

earthengine authenticate

# Inspiration: https://climada-python.readthedocs.io/en/stable/tutorial/climada_util_earth_engine.html

# Amanda Boatswain Jacques / Etienne Lord

# Since 20 March 2021  
# Last updated : 2021-04-22 5:55 PM 
"""

# Library imports
import pandas as pd
from matplotlib import pyplot as plt

# Import the .csv file as a pandas dataframe 
dataset = pd.read_csv("points_ALL_categories_2018_ACI.csv")
year = 2018

# Keep only the data that is available and drop the columns we don't need
dataset = dataset.drop(["Unnamed: 0", "Region", "ACI Crop ID", "File Name"], axis =1)
dataset = dataset[dataset["Available"] == True]
print(dataset.head()) 
print(" ")

# Set the point id as the index 
point_grouped_data = dataset.groupby("Point ID").agg({"Point ID": ["count"]})
point_counts = point_grouped_data["Point ID"].apply(pd.value_counts).sort_index()

month_grouped_data = dataset.groupby(["Month Start Date"]).agg({"Month Start Date": ["count"]})
month_counts = month_grouped_data["Month Start Date"].sort_index()

crop_grouped_data = dataset.groupby(["Crop Type"]).agg({"Crop Type": ["count"]}).sort_index()
crop_counts = crop_grouped_data["Crop Type"].sort_index()

crop_point_grouped_data = dataset.groupby(["Crop Type", "Point ID"]).count()
#crop_point_grouped_data = crop_point_grouped_data.agg({"count": ["count"]})

prov_grouped_data = dataset.groupby(["Province Code"]).agg({"Province Code": ["count"]}).sort_index()
prov_counts = prov_grouped_data["Province Code"].sort_index()

# Create some histogram plots of the data to visualize the distributions 

# Check how many images have how many available points
ax = point_counts.plot.bar(figsize=(10, 6)) 

# Display the points counts
print("Point Counts: ", point_counts.head())
print(" ")

# clean up the figure a bit
ax.set_title("Dataset image distribution " + "(" + str(year) + ")")
ax.set_ylabel("Counts")
ax.set_xlabel("Number of Available Images")
ax.set_xticklabels(["1", "2", "3 ", "4", "5"], rotation = 0)
plt.tight_layout()
plt.show()

# Check how many images are available per month 
ax2 = month_counts.plot.bar(figsize=(10, 6)) 

# Display the month counts
print("Month Counts: ", month_counts.head())
print(" ")

# clean up the figure a bit
ax2.set_title("Number of images available per month " + "(" + str(year) + ")")
ax2.set_ylabel("Counts")
ax2.set_xlabel("Month")
ax2.set_xticklabels(["June", "July", "August", "September", "October"], rotation = 0)
plt.tight_layout()
plt.show()


# Check how many images are available per crop type 
ax3 = crop_counts.plot.bar(figsize=(10, 6)) 

# Display the crop counts
print("Crop Counts: ", crop_counts.head())
print(" ")

# clean up the figure a bit
ax3.set_title("Number of images available per crop type " + "(" + str(year) + ")")
ax3.set_ylabel("Counts")
ax3.set_xlabel("Crop Type")
#ax3.set_xticklabels(["Barley", "Corn", "Mixed Wood", "Pasture"], rotation = 0)
plt.tight_layout()
plt.show()


# Check how many images are available per Province 
ax4 = prov_counts.plot.bar(figsize=(10, 6)) 

# Display the crop counts
print("Province Counts: ", prov_counts.head())
print(" ")

# clean up the figure a bit
ax4.set_title("Number of images available per province "  + "(" + str(year) + ")")
ax4.set_ylabel("Counts")
ax4.set_xlabel("Province Code")
#ax3.set_xticklabels(["Barley", "Corn", "Mixed Wood", "Pasture"], rotation = 0)
plt.tight_layout()
plt.show()






"""
# Visualize the longitude, latitude
fig, _ = plt.subplots(1, 2, figsize=(10, 6))
dataset["Longitude"

ax4 = dataset[["Longitude", "Latitude"]].plot.box(subplots=True,  figsize=(10, 6)) 
print(" ")

# clean up the figure a bit
ax4.set_title("Longitude and latitude distribution")
ax4.set_ylabel("Value (Decimal Degrees)")
ax4.set_xlabel("Coordinates")
ax4.set_xticklabels(["Longitude", "Latitude"], rotation = 0)
plt.tight_layout()
plt.show()
"""
