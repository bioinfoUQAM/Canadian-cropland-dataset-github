# Usage python create_training.py --indir directory

# Divide directory into sample directory
import argparse
import numpy as np
import os
import shutil
from pathlib import Path
import utils


#directory = "MILLET"
directory = "OAT"
#directory = "POTATO"


imagePaths = utils.get_all_file_paths(directory)
print("Number of files: ", len(imagePaths))
 
#print("imagePaths ", imagePaths)
ids = list(set([("POINT_" + imagePath.split("_")[1] +"_") for imagePath in imagePaths]))
#print('ids :', ids)

print("Number of unique ids: ", len(ids))

def check_sequence_order(sequence):
    dates = [int(item.split("_")[2]) for item in sequence] 
    print("dates", dates)
    print("Result",  (np.diff([dates])) )
    if np.all(np.diff([dates]) == 1):
        print("Valid sequence found!")
        return sequence
    else: 
        print("Invalid sequence!")

def construct_set(set_type, imagePaths):
    new_set = []
    for set_index in set_type:
        #print("Set index: ", set_index)
        matches = list(filter(lambda x: set_index in x, imagePaths))      
        print("Number of images found: ", len(matches))
        matches.sort()
        #print("Matches: ", matches)
        new_set.append(matches)
        
    return new_set


sets = construct_set(ids, imagePaths)
#print(sets)

image_sequences = []
for group in sets: 
    print("original group \n", group)
    n = len(group)
    if n > 2:  # if the set only has 2 images or less, ignore it
        if n%3 > 0:
            print("n", n)
            print("n%3", n%3)
            for i in range(0, n%3 + 1):
                print("i", i)
               
                #print(group)
                sequence =  group[i:i+3] 
                print("items", sequence) 
                """ do a check here to see if the list is good """
                check_sequence_order(sequence)
                
                
                #for item in group:
                #print(item)
                image_sequences.append(sequence)
    
        else: 
            pass
    print(" ")




"""
def create_set(datadir, ptrain, pval, seed, number):
  print("Creating train/val/test directories for: ", datadir) 
  PATH = datadir
  imagePaths = utils.get_all_file_paths(datadir)
  files = imagePaths
  print("Number of files: ", len(files))
 
  #print("imagePaths ", imagePaths)
  ids = list(set([("POINT_" + imagePath.split("_")[1] +"_") for imagePath in imagePaths]))
  #print('ids :', ids)
  
  print(len(files))
  np.random.seed(seed)
  np.random.shuffle(ids)
  
  if number>-1: 
      ids = ids[:number]
 
  (training, validation, test) = divides(ids, ptrain, pval)
  print(" ")
  print("training ids: ",   sorted(training))
  print(" ")
  
  print("validation ids: ", sorted(validation))
  print(" ")
  print("test ids: ", sorted(test))
  
  print(" ")
  
  training_files = construct_set(training, imagePaths)
  print("Number of training files: ", len(training_files))
  #for file in training_files:
      #print(file)
  #print(" ")
  
  validation_files = construct_set(validation, imagePaths)
  print("Number of validation files: ", len(validation_files))
  #print(validation_files)
  #for file in validation_files:
      #print(file)
  #print(" ")
  
  test_files = construct_set(test, imagePaths)
  print("Number of test files: ", len(test_files))
  #for file in test_files:
      #print(file)
  #print(" ")

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
    shutil.copy2(f,str(PATH)+"_test"+"/"+pth+"/"+filename)
    
  for f in validation_files:
    p = Path(f)
    filename=p.parts[-1]
    pth=p.parts[-2]
    os.makedirs(str(PATH)+"_validation"+"/"+pth, exist_ok=True)
    shutil.copy2(f,str(PATH)+"_validation"+"/"+pth+"/"+filename)
    
  for f in training_files:
    p = Path(f)
    filename=p.parts[-1]
    pth=p.parts[-2]
    os.makedirs(str(PATH)+"_training"+"/"+pth, exist_ok=True)
    shutil.copy2(f,str(PATH)+"_training"+"/"+pth+"/"+filename)

"""

#create_set(args.indir, args.ptrain, args.pval, args.seed, args.n)