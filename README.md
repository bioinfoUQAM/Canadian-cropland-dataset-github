# Canadian-Cropland-Dataset
This repository houses a novel patch-based dataset compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2.

## Dataset Description
This repository contains code for the creation and manipulation of a novel patch-based dataset, the _Canadian Cropland Dataset_ inspired by the [Eurosat dataset](https://ieeexplore.ieee.org/document/8736785 "Eurosat article"). It is compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2. A total of 58,973 high-resolution (10 m/pixel) geo-referenced images of 46 crop types over 5 months and 4 years were extracted using the [Google Earth Engine](https://earthengine.google.com/  "Google Earth Engine") (GEE) and were automatically labelled with the [Canadian Crop Inventory](https://www.agr.gc.ca/atlas/aci "Canadian Crop Inventory"). Each image contains 12 main spectral bands as well as a selection of bands corresponding to vegetation indices (GNDVI, NDVI, NDVI45, OSAVI and PSRI). 

The original dataset can be accessed through this [Google Drive](https://drive.google.com/drive/folders/1mNI8B5EMk0Xgvx2Pc9ztnQRaW9pXh8yb?usp=sharing "Link to dataset") link. In the upcoming months, we will be hosting it on a website at Université du Québec à Montreal. The Drive contains the images from all years preprocessed at multiple times. A version of the dataset in train/validation/test splits is available for performing machine learning benchmarking tests and can be found in the folder "Divided_datasets_2016_2019". The folder "Overview_of_dataset_2016_2019" is a glossary showing all the images available over the four year period for each of the 4,982 geographical points.

![dataset overview](https://github.com/bioinfoUQAM/AAFC-cropland-dataset/blob/main/figures/crop_type_mosaic.png)

Figure 1: An overview of sample patches of some of the crop classes in the proposed dataset. The images measure 64 x 64 pixels and have a spatial resolution of 10m/pixel. 

## Python version
* [python 3.8.8](https://www.python.org/downloads/release/python-388/)

## Other libraries
You can install these required libraries using the `conda install -c conda-forge --library_name` command:

```
conda install -c conda-forge earthengine-api
conda install -c conda-forge keras-gpu  
conda install -c conda-forge imutils
conda install -c conda-forge scikit-learn
conda install -c conda-forge scikit-image
conda install -c conda-forge numpy
conda install -c conda-forge opencv
conda install -c conda-forge pandas
conda install -c conda-forge pillow
conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
conda install -c conda-forge rasterio
```

## Description of Repository

### Data Cleaning
Contains functions for manipulating images and moving files around to create label specific directories and training/validation/test sets. We use a JavaScript application for removing cloudy/noisy images from the collection (NOT CONTAINED IN THIS REPOSITORY). 

![rapid_tags](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/figures/rapid_tags.png)

The dataset is manually curated by removing: 
- Cloudy images
- Noisy images
- Images with missing pixels

### Data Collection
Contains multiple python scripts for downloading the Sentinel-2 images for each point in the sample .csv file. It also contains some data visualization code. 

### Dataset Statistics
Contains spreadsheets and figures depicting the distribution of the images within the dataset. 

### Earth Engine
Contains the [JavaScript code](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/EarthEngine/AAFC_GEE_points.js "JavaScript code") used to collect points of agricultural fields all accross Canada from the months of June 2016 to October 2019. The code can be visualized directly in GEE using this [link](https://code.earthengine.google.com/?scriptPath=users%2Famandaboatswainj%2FAAFC-cropland-database%3AAAFC_GEE_dataset_points "link"). Note that you must be registered with an activated GEE account to view the script and run it on the cloud.
 
![geographical points](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/figures/ACI_crop_inventory_2019.png)
Figure 2: Map representing an overview of the selected geographical locations used in the _Canadian Cropland Dataset_. Markers are randomly chosen fields and are color-coded by the 2019 crop types.

### Machine Learning
Contains python code for the deep learning benchmark models (CNN, ResNet-50 and LRCN).

### References
If you have used this dataset, please cite the following presentation:

[1] Towards the Creation of a Canadian Land-Use Dataset for Agricultural Land Classification. Amanda A. Boatswain Jacques, Abdoulaye Baniré Diallo, Etienne Lord.  42nd Canadian Symposium on Remote Sensing: Understanding Our World: Remote Sensing for a Sustainable Future, June 2021. Virtual. 


## Contact
For any questions or concerns regarding this repository, please contact boatswain_jacques.amanda@courrier.uqam.ca. 

