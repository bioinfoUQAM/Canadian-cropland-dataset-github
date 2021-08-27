# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:49:46 2021

@author: Amanda
"""

import os 
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# Get all the RGB file paths for every year 

directory = "../AAFC-dataset/full/"


# initializing empty file paths list
file_paths = []
  
# crawling through directory and subdirectories
for root, directories, files in os.walk(directory):
    for filename in files:
        # join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        if "RGB" in filepath:
            file_paths.append(filepath)
  
# return all file paths in a list

d = []
for file in file_paths:
    year =  file.split("\\")[0].split("/")[-1]
    _ , _ , train_set, class_name, filename =  file.split("\\")     
    _, point_ID, month, province, *_ =  filename.split("_")

     
    d.append(
        {
            'YEAR': year,
            'YEAR/MONTH': month,
            'MONTH': month[-2:],
            'POINT ID': point_ID,
            'PROVINCE': province,
            'CLASSNAME': class_name, 
            'FILENAME': filename
        }
    )



df = pd.DataFrame(d)


# Set the point id as the index 
point_grouped_data = df.groupby("POINT ID").agg({"POINT ID": ["count"]})
point_counts = point_grouped_data["POINT ID"].apply(pd.value_counts).sort_index()

point_grouped_year_data = df.groupby(["YEAR", "POINT ID"]).agg({"POINT ID": ["count"]})
#counts_2019 = point_grouped_year_data[('POINT ID', 'count')]["2019"].groupby(["POINT ID", 'count']).agg({"POINT ID": ["count"]})

month_grouped_data = df.groupby(["MONTH"]).agg({"MONTH": ["count"]})
month_counts = month_grouped_data["MONTH"].sort_index()

crop_grouped_data = df.groupby(["CLASSNAME"]).agg({"CLASSNAME": ["count"]}).sort_index()
crop_counts = crop_grouped_data["CLASSNAME"].sort_index()

crop_point_grouped_data = df.groupby(["CLASSNAME", "POINT ID"]).count()
#crop_point_grouped_data = crop_point_grouped_data.agg({"count": ["count"]})

prov_grouped_data = df.groupby(["PROVINCE"]).agg({"PROVINCE": ["count"]}).sort_index()
prov_counts = prov_grouped_data["PROVINCE"].sort_index()

# Check how many images have how many available points
#ax = point_counts.plot.bar(figsize=(12,5)) 


df_2016 = df[df["YEAR"] == "2016"]
df_2017 = df[df["YEAR"] == "2017"]
df_2018 = df[df["YEAR"] == "2018"]
df_2019 = df[df["YEAR"] == "2019"]


# Plot all years at once (image counts)
point_2016_counts = df_2016.groupby("POINT ID").agg({"POINT ID": ["count"]}).apply(pd.value_counts).sort_index()
point_2017_counts = df_2017.groupby("POINT ID").agg({"POINT ID": ["count"]}).apply(pd.value_counts).sort_index()
point_2018_counts = df_2018.groupby("POINT ID").agg({"POINT ID": ["count"]}).apply(pd.value_counts).sort_index()
point_2019_counts = df_2019.groupby("POINT ID").agg({"POINT ID": ["count"]}).apply(pd.value_counts).sort_index() 


all_counts = [point_2016_counts, point_2017_counts, point_2018_counts, point_2019_counts]
all_counts = pd.concat(all_counts, axis = 1)


# Plot all years at once (class counts)
point_2016_classes = df_2016.groupby("CLASSNAME").agg({"CLASSNAME": ["count"]}).sort_index()
point_2017_classes = df_2017.groupby("CLASSNAME").agg({"CLASSNAME": ["count"]}).sort_index()
point_2018_classes = df_2018.groupby("CLASSNAME").agg({"CLASSNAME": ["count"]}).sort_index()
point_2019_classes = df_2019.groupby("CLASSNAME").agg({"CLASSNAME": ["count"]}).sort_index() 

all_classes = [point_2016_classes, point_2017_classes, point_2018_classes, point_2019_classes]
all_classes = pd.concat(all_classes, axis = 1)


# Plot all years at once (month counts)
point_2016_months = df_2016.groupby("MONTH").agg({"MONTH": ["count"]}).sort_index()
point_2017_months = df_2017.groupby("MONTH").agg({"MONTH": ["count"]}).sort_index()
point_2018_months = df_2018.groupby("MONTH").agg({"MONTH": ["count"]}).sort_index()
point_2019_months = df_2019.groupby("MONTH").agg({"MONTH": ["count"]}).sort_index() 

all_months = [point_2016_months, point_2017_months, point_2018_months, point_2019_months]
all_months = pd.concat(all_months, axis = 1)

# Plot all years at once (province counts)
point_2016_provinces = df_2016.groupby("PROVINCE").agg({"PROVINCE": ["count"]}).sort_index()
point_2017_provinces = df_2017.groupby("PROVINCE").agg({"PROVINCE": ["count"]}).sort_index()
point_2018_provinces = df_2018.groupby("PROVINCE").agg({"PROVINCE": ["count"]}).sort_index()
point_2019_provinces = df_2019.groupby("PROVINCE").agg({"PROVINCE": ["count"]}).sort_index() 

all_provinces = [point_2016_provinces, point_2017_provinces, point_2018_provinces, point_2019_provinces]
all_provinces = pd.concat(all_provinces, axis = 1)

patterns =('o', '+', 'x','/')

"""CREATE THE MIXED PLOTS""" 

# POINT COUNTS

# Display the points counts
print("All years (seperate) points counts: ", all_counts)
print(" ")

ax = all_counts.plot.bar(figsize=(12,7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit

ax.set_ylabel("Number of sets", fontsize=15)
ax.set_xlabel("Images in a set", fontsize=15)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(10)

ax.yaxis.labelpad = 10
ax.xaxis.labelpad = 10

bars = ax.patches
hatches = [p for p in patterns for i in range(len(all_counts))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

plt.ylim([0, 2500])
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(fontsize=12) # using a size in points
plt.margins(x=0, y=0)
plt.legend(["2016", "2017", "2018", "2019"], fontsize=12, loc='upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax.get_xticklabels(), ha="right", rotation = 360)



# CLASSES COUNTS

# Display the class counts
print("Number of images per class (all years seperate): ", all_classes)
print(" ")

# Check how many images are available per class 
ax2 = all_classes.plot.bar(figsize=(14, 6), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax2.set_ylabel("Number of images", fontsize=15)
ax2.set_xlabel("Crop type", fontsize=15)

for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	label.set_fontsize(10)

ax2.yaxis.labelpad = 10
ax2.xaxis.labelpad = 10

bars = ax2.patches
hatches = [p for p in patterns for i in range(len(all_classes))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["2016", "2017", "2018", "2019"], fontsize=12, loc='upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax2.get_xticklabels(), ha="right", rotation=45)


# MONTHS COUNTS

# Display the months counts
print("Number of images per month (all years seperate): ", all_months)
print(" ")

# Check how many images are available per class 
ax3 = all_months.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax3.set_ylabel("Number of images", fontsize=15)
ax3.set_xlabel("Month", fontsize=15)

for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
	label.set_fontsize(12)

ax3.yaxis.labelpad = 10
ax3.xaxis.labelpad = 10

bars = ax3.patches
hatches = [p for p in patterns for i in range(len(all_months))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["2016", "2017", "2018", "2019"], fontsize=12, loc='upper left') # using a size in points
plt.tight_layout()
ax3.set_xticklabels(["June", "July", "August", "September", "October"], rotation = 0)
plt.plot()


# PROVINCE COUNTS

# Display the months counts
print("Number of images per province (all years seperate): ", all_provinces)
print(" ")

# Check how many images are available per class 
ax4 = all_provinces.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax4.set_ylabel("Number of images", fontsize=15)
ax4.set_xlabel("Province code", fontsize=15)

for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
	label.set_fontsize(15)

ax4.yaxis.labelpad = 10
ax4.xaxis.labelpad = 10

bars = ax4.patches
hatches = [p for p in patterns for i in range(len(all_provinces))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)


plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["2016", "2017", "2018", "2019"], fontsize=12, loc='upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax4.get_xticklabels(), rotation=360)


""" PLOTTING ALL DATA COMBINED """

# Display the points counts
print("All years (combined) points counts: ", point_counts)
print(" ")

# Check how many images are available per crop type 
ax5 = point_counts.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax5.set_ylabel("Number of sets", fontsize=15)
ax5.set_xlabel("Images in a set", fontsize=15)

for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
	label.set_fontsize(15)

ax5.yaxis.labelpad = 10
ax5.xaxis.labelpad = 10
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["Count"], fontsize=15,  loc = 'upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax5.get_xticklabels(), ha="right", rotation=0)


# CLASSES COUNTS
# Display the classes counts
print("All years (combined) classes counts: ", crop_counts)
print(" ")

# Check how many images are available per crop type 
ax6 = crop_counts.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax6.set_ylabel("Number of images", fontsize=15)
ax6.set_xlabel("Crop type", fontsize=15)

for label in (ax6.get_xticklabels() + ax6.get_yticklabels()):
	label.set_fontsize(12)

ax6.yaxis.labelpad = 10
ax6.xaxis.labelpad = 10
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["Count"], fontsize=15,  loc = 'upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax6.get_xticklabels(), ha="right", rotation=45)


# MONTHS COUNTS
# Display the months counts
print("All years (combined) months counts: ", month_counts)
print(" ")

# Check how many images are available per crop type 
ax7 = month_counts.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax7.set_ylabel("Number of images", fontsize=15)
ax7.set_xlabel("Month", fontsize=15)

for label in (ax7.get_xticklabels() + ax7.get_yticklabels()):
	label.set_fontsize(12)

ax7.yaxis.labelpad = 10
ax7.xaxis.labelpad = 10
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["Count"], fontsize=15,  loc = 'upper left') # using a size in points
ax7.set_xticklabels(["June", "July", "August", "September", "October"], rotation = 0)
plt.tight_layout()
plt.plot()



# PROVINCE COUNTS
# Display the province counts
print("All years (combined) months counts: ", prov_counts)
print(" ")

# Check how mny images are available per crop type 
ax8 = prov_counts.plot.bar(figsize=(12, 7), cmap = mpl.cm.get_cmap('PiYG')) 

# clean up the figure a bit
ax8.set_ylabel("Number of images", fontsize=15)
ax8.set_xlabel("Province code", fontsize=15)

for label in (ax8.get_xticklabels() + ax8.get_yticklabels()):
	label.set_fontsize(12)

ax8.yaxis.labelpad = 10
ax8.xaxis.labelpad = 10
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.6, top=0.8)
plt.legend(["Count"], fontsize=15,  loc = 'upper left') # using a size in points
plt.tight_layout()
plt.plot()
plt.setp(ax8.get_xticklabels(), ha="right", rotation=0)


