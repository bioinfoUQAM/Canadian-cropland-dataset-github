# -*- coding: utf-8 -*-
"""
Modified on Aug 11 2021

A set of helper functions for manipulating data for the AAFC cropland dataset
@author: Amanda A. Boatswain Jacques
"""

# import the necessary libraries
import numpy as np
import cv2
import re
import os
import shutil
from skimage.exposure import is_low_contrast
from sklearn.preprocessing import LabelEncoder
from tifffile import tifffile
import zipfile



""" FILE MANIPULATION """

# unzip a file in a user-specified location
def unzip_file (zip_file, output_directory):
    """
    Parameters
    ----------
    zip_file : .ZIP
        A .zip file that needs to be unzipped.
        
    output_directory : STRING
        The desired output location of the unzipped file.

    Returns
    -------
    None.
    """
    # unzip the zip dataset 
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_directory)
        
        
# move a group of files from one directory to another depending on the file extension.        
def move_files(start_path, end_path, ending):
    """
    Parameters
    ----------
    start_path : STRING
        Initial directory.
        
    end_path : STRING
        End directory.
        
    ending : STRING
        File extension.

    Returns
    -------
    None.
    """
    for dirname, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            source = os.path.join(dirname, filename)
            
            # move the file to a specific directory depending on the file type 
            if filename.endswith(ending):
               shutil.move(source, end_path)
               print("File {} moved from {} to {}.".format(filename, start_path, end_path))
               
   
# convert a .tif to a readable grayscale image            
def convert_to_gray(imagePath):
    """
    Parameters
    ----------
    imagePath : STRING
        Path leading to a .tif image.

    Returns
    -------
    image : 1D NUMPY ARRAY
        Convert the tif image to a uint8 numpy array ranging from 0 to 255.
    """
    image = tifffile.imread(imagePath)*255
    image = np.uint8(image)
    
    return image
               

# merge the red, green and blue bands to create a 3D images. Must use convert_to_gray
# on the bands first. 
def convert_to_BGR(blue, green, red, filename = None, save = False):
    """
    Parameters
    ----------
    blue :  1D NUMPY ARRAY
        Blue pixel channel.
        
    green :  1D NUMPY ARRAY
        Green pixel channel.
        
    red :  1D NUMPY ARRAY
        red pixel channel.

    Returns
    -------
    rgb:  3D NUMPY ARRAY
        Convert the 3 color channels to a 3D BGR image. Save the image as a 
        .png file if requested. 
    """
    # Stack the bands accross the 3rd dimension
    bgr = np.dstack((blue, green, red))
    
    # save the resulting RGB image as a .png
    if save == True: 
        cv2.imwrite(filename + ".png", bgr)
    
    return bgr


# Check whether an image has enough pixel variation in it to be considered useful 
def check_contrast(image, threshold = 0.15): 
    """
    Parameters
    ----------
    image :  1D NUMPY ARRAY
      
    threshold : FLOAT, optional
        An image is considered low-contrast when its range of brightness spans
        less than the fraction of its data type's full range specified by 
        threshold. The default is 0.15.

    Returns
    -------
    contrast : BOOLEAN
    """
    # check to see if the image is low contrast
    contrast = is_low_contrast(image, fraction_threshold=threshold)
    
    if contrast == True: 
        print("[WARNING] Image is low contrast. Consider removing this image.")
    
    return contrast 


# Get all the file paths in a given directory
def get_all_file_paths(directory):

    """ Parameters
        ----------
        directory (STRING): directory of files

        Returns
        -------
        file_paths (LIST): list of all the files in that directory
    """

    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath and add to list
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


# Create sets of image files belonging to each point
def create_image_sets(directory):

    """ Parameters
        ----------
        directory (STRING) : directory of files

        Returns
        -------
        new_set (SET) : A list of lists, where each item has all of the files from a unique point
    """
    imagePaths = get_all_file_paths(directory)
    # get the point IDs
    ids = list(set([("POINT_" + imagePath.split("_")[1] +"_") for imagePath in imagePaths]))
    new_set = []

    for set_index in ids:
        matches = list(filter(lambda x: set_index in x, imagePaths))
        matches.sort() # sort chronologically
        new_set.append(matches)
        
    # return the filtered image sets
    return new_set


def get_all_set_images(set_type, imagePaths):
    
    """ Parameters
        ---------- 
        set_type (SET) : a set of point ids (i.e: train, val, test ids)
        imagePaths (LIST) : list of strings leading to the images in a directory
    """
    new_set = []
    for set_index in set_type:
        #print(set_index)

        # filter list to only keep the images that are in either the train, val, test set
        matches = list(filter(lambda x: set_index in x, imagePaths))
        #print(matches)

        for match in matches:
            new_set.append(match)

    # return only the filtered images
    return new_set


# Get the dates from the image series and make sure they follow each other by one month only
def is_valid_sequence(sequence, seq_len):

    """ Parameters
        ----------
        sequence (LIST) : an item in the list generated by construct_sets()
        seq_len (INT) : number of items in an image sequence

        Returns
        -------
        BOOLEAN
    """
    # get year values in a sequence
    dates = [int(item.split("_")[2]) for item in sequence]

    if len(dates) == seq_len and np.all(np.diff([dates]) == 1):
        #print("Valid sequence found!")
        return True
    else:
        return False


# Count the number of valid sequences in a set
def get_valid_sequence_count(imageSet, seq_len):

    """ Parameters
        ----------
        imageSet (SET): set of images generated by create_image_sets()
        seq_len (INT) : number of items in an image sequence

        Returns
        ---------
        count (INT): number of OK sets
    """
    count = 0
    for group in imageSet:
        n = len(group)

        for i in range(0, n % seq_len + 1):
            # get the first n (seq_len) elements in the list
            sequence =  group[i:i+seq_len]

            # do a check here to see if the list is good, then return it
            if is_valid_sequence(sequence, seq_len):
                count += 1

    return count


""" TRAINING FUNCTIONS """

# DATA, % Training, % validation
# Divide the training data into a training, validation and test set
def divides(data, ptraining = 0.70, pvalidation = 0.15):
    """
    :param data: (SET) python set of all unique points
    :param ptraining: (FLOAT) fraction of images allocated to training default is 0.70.
    :param validation: (FLOAT)  fraction of images allocated to validation default is 0.15.
    Remainder of images will be set aside for testing.
    """
    ptest = 1-(ptraining+pvalidation)
    l=len(data)
    print("Number of unique points: ", l)

    # calculate the number of images in each set
    tq=int((l)*pvalidation)
    tend=int((l) *ptraining)

    # find the starting and end points in the list for the training, validation and test sets
    vstart=tend
    vend=vstart+tq
    teststart=vend

    # split the data into 3 seperate lists
    ret=[data[:tend], data[vstart:vend], data[teststart:]]

    print("Training  : " + str(len(ret[0]))+" ("+str(ptraining*100)+" %)")
    print("Validation: " + str(len(ret[1]))+" ("+str(pvalidation*100)+" %)")
    print("Test      : " + str(len(ret[2]))+" ("+ "{:2.1f}".format(ptest*100) +" %)")

    return (ret)


# create a set with all the unique labels
def create_labels_set(directory, imagePaths):

    """ Parameters
        ----------
        imagePaths (LIST) : list of images generated by get_all_file_paths()

        Returns
        -------
        labels (SET) : set of all existing labels/classes
    """

    labels = set()
    for file in imagePaths:
        label = get_classname(directory, file)
        labels.add(label)

    return labels


# change string label(s) to numerial classes
def encode_label(binarizer, label):
    
    """ Parameters
        ----------
        binarizer (OBJECT): label binarizer object
        label (STRING) : classname 
        
        Returns
        -------
        label (INT)

    """
    label = binarizer.transform(label)
    return label


# create a label binarizer object
def create_binarizer(labels):
    
    lb = LabelEncoder()
    lb.fit(list(labels))
    return lb


# Find the classname from an image following the file pattern
def get_classname(directory: str, imagePath: str) -> str:
    # (taken from keras_video library)

    """ Parameters
        ----------
        directory (STRING) : directory of files (training, validation or test)
        imagePath (STRING) : path to the image

        Returns
        ------
        classname (STRING)
    """

    # create the glob pattern
    glob_pattern = "./" + directory + "/{classname}/*.png"

    # work with real path
    imagePath = os.path.realpath(imagePath)

    pattern = os.path.realpath(glob_pattern,)

    # remove special regexp chars
    pattern = re.escape(pattern)

    # get back "*" to make it ".*" in regexp
    pattern = pattern.replace('\\*', '.*')

    # use {classname} as a capture
    pattern = pattern.replace('\\{classname\\}', '(.*?)')

    # and find all occurence
    classname = re.findall(pattern, imagePath)[0]
    return classname


# Generator -> generates a sequence of images of a specified length
def image_sequence_generator(imageSet, seq_len, mode = "train"):

    """ Parameters
        ----------
        imageSet (SET): set of images generated by create_image_sets()
        seq_len (INT) : number of items in an image sequence
        mode (STRING) : options are "train" or "eval". If "eval", the generator will stop before looping indefinitely.

        Yields
        ------
        sequence (LIST) : list of file names used to create a training instance
    """
    
    while True:
        for group in imageSet:
            n = len(group)

            # if the set only has 2 images or less, ignore it
            if n > 2 and n % seq_len >= 0:
                for i in range(0, n % seq_len + 1):
                    # get the first n (seq_len) elements in the list
                    sequence =  group[i:i+seq_len]
                    # do a check here to see if the list is good, then return it
                    if is_valid_sequence(sequence, seq_len):
                        yield sequence
                        
        if mode == "eval":
            break

# generate sets of images for training
def image_batch_generator(directory, binarizer, batch_size, input_shape, num_classes, seq_len=3, mode = "train"):

    """ Parameters
        ----------
        directory (STRING) : image directory (trainig/validation/test)
        binarizer (OBJECT): label binarizer
        batch_size (INT)
        input_shape (TUPLE) : image shape (width, height, channels) (only works with 3-D images for now)
        num_classes (INT) : number of classes
        seq_len (INT), optional : number of required images in a sequence. The default is 3 (only 3 is accepted).

        Yields
        -------
        batch_x (NUMPY ARRAY): an array of size (batch_size, seq_len, width, height, channels)
        batch_y (NUMPY ARRAY): an array of size (batch_size, num_classes) (one-hot encoded labels)
    """
    # create the sequence and image generators
    sets = create_image_sets(directory)
    image_gen = image_sequence_generator(sets, 3, mode)

    # initialize batch arrays
    x_shape = (batch_size, seq_len,) + input_shape
    y_shape = (batch_size, num_classes)

    while True:
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            #print("i: ", i)
            batch = next(image_gen)
            #print("Sequence length: ", len(batch))
            #print("Current batch: ", batch)

            # get the image paths in the set
            img1, img2, img3 = batch

            # get the class name of the series and encode it
            label1 = get_classname(directory, img1)
            label1 = encode_label(binarizer, (label1,))

            # read the images in the set and store them

            img1 = cv2.imread(img1)
            img2 = cv2.imread(img2)
            img3 = cv2.imread(img3)

            """ For visualization """
            #result_img = cv2.hconcat([img1, img2, img3])
            #cv2.imshow("Loaded images", result_img)
            #cv2.waitKey(0)

            # append the images and the assigned label in a one-hot encoded vector
            image_batch = np.array([img1, img2, img3]) 

            # normalize the pixel values
            #image_batch.astype('float32')
            #image_batch = image_batch/255.0
            batch_x[i] = image_batch
            batch_y[i, label1] = 1


        yield (batch_x, batch_y)
