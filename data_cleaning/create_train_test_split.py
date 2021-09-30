# -*- coding: utf-8 -*-

# Usage python create_train_test_split.py -indir 2018/RGB/ -year 2018


"""
Created on Mon Aug  9 12:54:41 2021

@author: Amanda A. Boatswain Jacques
"""

# import the necessary libraries
import argparse
import numpy as np
import os
import utils as utils
import pickle

# create the argument parser and parse the args 
parser = argparse.ArgumentParser(description='Create files to split the training, \
                                 validation and test data using points. Images from \
                                the same point are separated.')
                                
parser.add_argument('-indir', type=str, help='input dir for data')
parser.add_argument('-year', type=int, help='image dataset year (for saving splits)')
parser.add_argument('-ptrain', type=int, help='percent training', default=0.70)
parser.add_argument('-pval', type=int, help='percent validation', default=0.15)
parser.add_argument('-seed', type=int, help='random seed for np.shuffle', default=42)

args = parser.parse_args()


# Capture the arguments from the command line 
ptrain, pval = args.ptrain, args.pval
datadir = args.indir 
year = str(args.year)
seed = args.seed


ALLimagePaths = utils.get_all_file_paths(datadir)
print("Number of Images available: ", len(ALLimagePaths))

# get all the class folders 
classes = os.listdir(datadir)


train_points = []
val_points = []
test_points = []

# go through each image class subfolder
for folder in classes:
    print("Folder: ", folder)
    imagePaths = utils.get_all_file_paths(datadir + "/" + folder)
    print("Number of images: ", len(imagePaths))
    ids = list(set([(imagePath.split("/")[-1].split("\\")[0] + ("\\") + "POINT_" + imagePath.split("_")[1] +"_") for imagePath in imagePaths]))
    np.random.seed(seed)
    np.random.shuffle(ids)
    
    (training, validation, test) = utils.divides(ids, ptrain, pval)
    train_points = train_points + training
    val_points = val_points + validation
    test_points = test_points + test

print("Number of training points :", len(train_points))
print("Number of validation points :", len(val_points))
print("Number of test points :", len(test_points))


train_file = "train_points_list_" + year + ".pkl"
val_file = "val_points_list_" + year + ".pkl"
test_file = "test_points_list_" + year + ".pkl"

filenames = (train_file, val_file, test_file)
set_names = (train_points, val_points, test_points)

open_train_file = open(train_file, "wb")
pickle.dump(train_points, open_train_file)
open_train_file.close()

open_val_file = open(val_file, "wb")
pickle.dump(val_points, open_val_file)
open_val_file.close()

open_test_file = open(test_file, "wb")
pickle.dump(test_points, open_test_file)
open_test_file.close()
