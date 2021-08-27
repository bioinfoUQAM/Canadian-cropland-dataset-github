# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:27:37 2021

@author: Amanda
"""

# USAGE  python download_sentinel_images.py -f suivi_champs.csv -t "2017-06-01"  "2017-06-30" -n "ID" "Point"

# import librairies
import argparse
from EarthEngineDownloader import EarthEngineDownloader 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filepath", required=True, help="path to the .csv with the GPS coordinates")
ap.add_argument("-t", "--time interval", nargs="*", type = str,  required=True, 
                help="time interval for which images are required")
ap.add_argument("-n", "--name fields",  nargs="*", type =str, default =[], required=True,
	help="list of fields in the .csv required for naming output folder")
args = vars(ap.parse_args())


EDD = EarthEngineDownloader()
EDD.download_images_from_file(args["filepath"], args["time interval"], args["name fields"])

