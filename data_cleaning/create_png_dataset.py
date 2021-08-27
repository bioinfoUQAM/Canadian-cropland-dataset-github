# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:41:09 2021
@author: amanda
"""

# loop through the .zip files and create images in .png format
# import necessary libraries

import os
import image_to_png

directory = "dataset_zip"

#file_extensions = ["OSAVI", "NDVI", "GNDVI", "PSRI", "NDVI45"]
extension = "OSAVI"
print("Extension: ", extension)

# crawling through directory and subdirectories
for root, directories, files in os.walk(directory):
    for filename in files:
        print("filname", filename)
        # join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        print("Filepath: ", filepath)

        """ For creating RGB images, no extension is required"""
        #image_to_png.RGB_spliter(filepath)

        image_to_png.three_channel_spliter(filepath, extension)
