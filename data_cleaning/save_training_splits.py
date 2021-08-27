# Usage python create_training.py --indir directory

# Divide directory into sample directory
#import argparse
import pickle
import numpy as np
import os
import shutil
from pathlib import Path
import utils


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

