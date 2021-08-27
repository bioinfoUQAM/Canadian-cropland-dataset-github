# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:41:33 2021

@author: amanda
"""

import os 
import shutil

# Directory containing all the images we would like to move around 
directory = "RGB"

# create the subdirectories to store the sorted files
crop_types = ['agriculture', 'barley', 'bean', 'blueberry', 'broadleaf', 'buckwheat',
       'canaryseed', 'canola', 'chickpea', 'coniferous', 'corn',
       'exposed_land', 'fababean', 'fallow', 'flaxseed', 'grassland', 'hemp',
       'lentil', 'millet', 'mixedwood', 'mustard', 'oat', 'orchard',
       'other_berry', 'other_vegetable', 'pasture', 'pea', 'potato', 'rye',
       'shrubland', 'soybean', 'spring_wheat', 'sugarbeet', 'sunflower',
       'triticale', 'unknown', 'urban', 'vineyard', 'water', 'wetland',
       'winter_wheat']

crop_types_caps = []

for crop in crop_types:
    directory_name = crop.upper()
    os.mkdir(directory_name)
    crop_types_caps.append(directory_name)

# crawling through directory and subdirectories
for root, directories, files in os.walk(directory):
    for filename in files:
        # join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        
        # move the file to the correct folder 
        for crop_name in crop_types_caps:
            if crop_name in filepath: 
                shutil.move(filename, crop_name)
                
            else:
                pass
        
  
