# AAFC-cropland-dataset
This repository houses a novel patch-based dataset compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2.

## Description
This repository contains code for the creation of a novel patch-based dataset, inspired by the [Eurosat dataset](https://ieeexplore.ieee.org/document/8736785 "Eurosat article"). It is compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2. A total of 58,973 high-resolution (10m) geo-referenced images of 46 crop types over 5 months and 4 years were extracted using the Google Earth Engine tool [Google Earth Engine](https://earthengine.google.com/  "GEE") and were automatically labelled with the [Canadian Crop Inventory](https://www.agr.gc.ca/atlas/aci "Canadian Crop Inventory").  The images contain 13 main spectral bands as well as a selection of bands corresponding to vegetation indices (GNDVI, NDVI, NDVI45, OSAVI and PSRI).

![dataset overview](https://github.com/bioinfoUQAM/AAFC-cropland-dataset/blob/main/figures/crop_type_mosaic.png)

### Python version
* [python 3.8.8](https://www.python.org/downloads/release/python-388/)

### Other libraries
You can install these required libraries using the `conda install -c conda-forge --library_name` command:

* earthengine-api
* keras-gpu  
* imutils
* scikit-learn
* scikit-image
* numpy
* opencv
* pandas
* pillow
* jupyterlab
* matplotlib
* rasterio

### Earth Engine
 Contains the [JavaScript code](https://github.com/AmandaBoatswain/AAFC-cropland-database/blob/main/EarthEngine/AAFC_GEE_points.js "JavaScript code") used to collect points of agricultural fields all accross Canada from the months of June 2016 to October 2019. The code can be visualized directly in GEE using this [link](https://code.earthengine.google.com/?scriptPath=users%2Famandaboatswainj%2FAAFC-cropland-database%3AAAFC_GEE_dataset_points "link" ). Note that you must be registered with an activated GEE account to view the script and run it on the cloud.

### Data Collection
Contains a python script for downloading the Sentinel-2 images for each point in the sample .csv file. It also contains some data visualization code. The original dataset is can be accessed through this Google Drive link. In the upcoming months, we will be hosting it on a website at Université du Quénec à Montreal.  Please contact authors for more inquiry. 

### Data Cleaning
Contains functions for manipulating images and moving files around to create label specific directories, as well as training, validation and test sets.

### Machine Learning
Contains a jupyter notebook for training the deep learning models (CNN and ResNet-50), as well as the saved models and tensorflow logs. 

## Contact
For any questions or concerns regarding this repository, please contact amanda.boatswainj@gmail.com

