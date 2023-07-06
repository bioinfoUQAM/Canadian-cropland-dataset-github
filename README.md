# The Canadian Cropland Dataset
This repository houses a novel patch-based dataset compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2.

## 1 - Dataset Description
This repository contains instructions and code for use with the novel patch-based dataset, the _Canadian Cropland Dataset_, inspired by the [Eurosat dataset](https://ieeexplore.ieee.org/document/8736785 "Eurosat article"). It is compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2. A total of 78,536 high-resolution geo-referenced images (Figure 1) of 10 main crop types over 5 months (June-October) and 4 years (2017-2020) were extracted using [Google Earth Engine](https://earthengine.google.com/  "Google Earth Engine") (GEE) and were automatically labelled with the [Canadian Crop Inventory](https://www.agr.gc.ca/atlas/aci "Canadian Crop Inventory"). Each image contains 12 main spectral bands as well as a selection of bands corresponding to vegetation indices (GNDVI, NDVI, NDVI45, OSAVI and PSRI). Images were collected using a list of 6,633 geographical points of Canadian agricultural fields. 

The dataset can be accessed through this [Google Drive](https://drive.google.com/drive/folders/1mNI8B5EMk0Xgvx2Pc9ztnQRaW9pXh8yb?usp=sharing "Link to dataset") link. In the upcoming months, we will be hosting it on a website at Université du Québec à Montreal. The Drive contains the preprocessed and cleaned images from all years in train/validation/test splits. Each split is identical and contains the same points across the image type (RGB, GNDVI, etc.). The paper related to this dataset can be found on ArXiv: https://arxiv.org/abs/2306.00114. 


![dataset overview](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/figures/crop_mosaic_10_categories.png)

Figure 1: An overview of sample patches of the crop classes in the dataset. The images measure 64 x 64 pixels and have a spatial resolution of 10 m/pixel. 

## 2 - Running the Software

### Python version
* [python 3.8.8](https://www.python.org/downloads/release/python-388/)

### Other libraries
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

## 3 - Description of Repository

### Data Cleaning
Contains functions for manipulating images and moving files around to create label specific directories and training/validation/test sets. We use a Java application for removing cloudy/noisy images from the collection (NOT CONTAINED IN THIS REPOSITORY). 

![rapid_tags](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/figures/rapid_tags.png)

Figure 2: A screenshot of the data cleaning software developed in Java. The dataset is manually curated by manually removing cloudy images, noisy images and images with missing pixels

### Data Collection
Contains multiple python scripts for downloading the Sentinel-2 images for each point in the sample .csv file. It also contains some data visualization code. 

### Dataset Statistics
Contains spreadsheets and figures depicting the distribution of the images within the dataset. 

### Earth Engine
Contains the [JavaScript code](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/earth_engine/CAN_GEE_points.js "JavaScript code") used to collect points of agricultural fields all accross Canada from the months of June 2017 to October 2020 (Figure 2). The code can be visualized directly in GEE using this [link](https://code.earthengine.google.com/?scriptPath=users%2Famandaboatswainj%2FAAFC-cropland-database%3AAAFC_GEE_dataset_points "link"). Note that you must be registered with an activated GEE account to view the script and run it on the cloud.
 
![geographical points](https://github.com/bioinfoUQAM/Canadian-cropland-dataset/blob/main/figures/ACI_point_map_2019.png)

Figure 3: Map representing an overview of the selected geographical locations used in the _Canadian Cropland Dataset_. Markers are randomly chosen fields and are color-coded by the 2019 crop types.

### Machine Learning
Contains python code for the deep learning benchmark models (ResNet-50, LRCN, 3D-CNN, etc).

## 4 - References
If you have used this dataset, please cite the following presentation:

[1] Towards the Creation of a Canadian Land-Use Dataset for Agricultural Land Classification. Amanda A. Boatswain Jacques, Abdoulaye Baniré Diallo, Etienne Lord.  42nd Canadian Symposium on Remote Sensing: Understanding Our World: Remote Sensing for a Sustainable Future, June 2021. Virtual. 


## 5 - Contact
For any questions or concerns regarding this repository, please contact boatswain_jacques.amanda@courrier.uqam.ca. 

