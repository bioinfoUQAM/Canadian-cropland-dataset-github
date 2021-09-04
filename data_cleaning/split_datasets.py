# Usage python split_datasets.py -indir ../dataset -outdirs GNDVI NDVI -year 2016

"""
Created on Mon Aug  9 12:54:41 2021

@author: Etienne Lord,  Amanda A. Boatswain Jacques
"""

import argparse
import shutil
import os
from pathlib import Path
import pickle
import utils as utils

# create the argument parser and parse the args
parser = argparse.ArgumentParser(description='Split the passed dataset into a \
                                 seperate directory using predetermined splits.')


parser.add_argument('-indir', type=str, help='input dir used to create splits for data')
parser.add_argument("-outdirs", required=True, nargs ="*", type=str, help="directories that need to be split")
parser.add_argument("-year", type=int, help="image dataset year (for loading splits)")


args = parser.parse_args()


# Capture the arguments from the command line
year = str(args.year)
indir = args.indir
PATHS =  [(indir +"/"+year+"/"+ directory+"/") for directory in args.outdirs]  # For RGB, GNDVI, NDVI, NDVI45, PSRI, OSAVI


# Load the train/val/test splits created by create_train_test_splits()
    # and seperate the images in the passed directories
train_file = "train_points_list_" + year + ".pkl"
val_file = "val_points_list_" + year + ".pkl"
test_file = "test_points_list_" + year + ".pkl"


# load the points from the pre-saved lists
open_train_file = open(train_file, "rb")
train_points = pickle.load(open_train_file)
open_train_file.close()

open_val_file = open(val_file, "rb")
val_points = pickle.load(open_val_file)
open_val_file.close()

open_test_file = open(test_file, "rb")
test_points = pickle.load(open_test_file)
open_test_file.close()


for PATH in PATHS:
    # Get all the file paths in that folder
    ALLimagePaths = utils.get_all_file_paths(PATH)

    print("Getting train/val/test images from %s " % PATH)
    # Get all the respective files from training, validation and testing.
    training_files = utils.get_all_set_images(train_points, ALLimagePaths)
    print("Number of training files: ", len(training_files))
    validation_files = utils.get_all_set_images(val_points, ALLimagePaths)
    print("Number of validation files: ", len(validation_files))
    test_files = utils.get_all_set_images(test_points, ALLimagePaths)
    print("Number of test files: ", len(test_files))
    print(" ")

    print("Moving files to _training, _validation and _test folders in %s. \n" % PATH)
    # Create the empty training, test, validation directories and move the files around
    shutil.rmtree(str(PATH)+"_training", ignore_errors=True)
    shutil.rmtree(str(PATH)+"_validation", ignore_errors=True)
    shutil.rmtree(str(PATH)+"_test", ignore_errors=True)

    os.makedirs(str(PATH)+"_training", exist_ok=True)
    os.makedirs(str(PATH)+"_validation", exist_ok=True)
    os.makedirs(str(PATH)+"_test", exist_ok=True)

    for f in test_files:
        p = Path(f)
        filename=p.parts[-1]
        pth=p.parts[-2]
        os.makedirs(str(PATH)+"_test"+"/"+pth, exist_ok=True)
        shutil.move(f,str(PATH)+"_test"+"/"+pth+"/"+filename)

    for f in validation_files:
        p = Path(f)
        filename=p.parts[-1]
        pth=p.parts[-2]
        os.makedirs(str(PATH)+"_validation"+"/"+pth, exist_ok=True)
        shutil.move(f,str(PATH)+"_validation"+"/"+pth+"/"+filename)

    for f in training_files:
        p = Path(f)
        filename=p.parts[-1]
        pth=p.parts[-2]
        os.makedirs(str(PATH)+"_training"+"/"+pth, exist_ok=True)
        shutil.move(f,str(PATH)+"_training"+"/"+pth+"/"+filename)
