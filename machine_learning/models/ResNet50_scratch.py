###############################################################################
# SETUP - EXECUTE THIS FIRST                                                  #
###############################################################################
# 1. Go into "Runtime" -> "Change runtime type" and Select "GPU" for hardward accelerator
# 2. Click the "Connect" button, at the right to start the instance.
# This will get the dataset into this instance
# 3. Note that the dataset RGB.zip, PSRI.zip, NDVI45.zip, NDVI.zip or GNDVI.zip must be in the root of your google drive
# 4. Name the dataset below before you start!
# 5. Use Tensorflow > 2.1 and keras_tuner 1.0.1 (pip install keras_tuner==1.0.1
###############################################################################
# Variables                                                                   #
###############################################################################
import os
dataset="RGB"  #Name the dataset here 
dataset_zip=dataset+".zip"
past_model="ResNetSC_RGB_"+dataset+"_*.hdf5"
final_model="ResNetSC_RGB_"+dataset+"_final.h5"

number_of_classes=10 
img_width, img_height = 64, 64
batch_size = 128
###############################################################################


###############################################################################
# HELPER FUNCTIONS
###############################################################################
def number_of_files(dirname):
	cpt = sum([len(files) for r, d, files in os.walk(dirname)])
	return cpt

################################################################################ 
# DÉFINITION DES DONNÉES D'ENTRÉE                                              #
################################################################################
train_data_dir = dataset+'/training'
validation_data_dir = dataset+'/validation'
test_data_dir = dataset+'/test'
nb_train_samples=number_of_files(train_data_dir)
nb_validation_samples=number_of_files(validation_data_dir)
nb_test_samples=number_of_files(test_data_dir)

# ==============================================================================
# Etienne Lord, Amanda Boatswain Jacques - 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# %tensorflow_version 1.x
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import kerastuner as kt
from kerastuner.applications import HyperResNet
from kerastuner import RandomSearch
import tensorflow as tf
from pathlib import Path
from ResNetConf import ResNet
import os

tf.random.set_seed(20210229)

################################################################################ 
# MODEL DEFINITION                                                             #
################################################################################

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


################################################################################ 
# IMAGES LOADING                                                               #
################################################################################

# Note, we could use data augmentation
train_datagen = ImageDataGenerator(rotation_range=90)
test_datagen = ImageDataGenerator() 


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
	  shuffle = True,
    class_mode='categorical') # Note: the class_mode is categorical

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
	shuffle = True,
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


################################################################################ 
# MODEL                                                                   #
################################################################################

hypermodel = ResNet(input_shape=input_shape, classes=number_of_classes)
model=hypermodel.build()
model.summary()

################################################################################ 
# RUN MODEL                                                                   #
################################################################################

csv_logger = CSVLogger("resnetSCPSRI3_"+dataset+"_last_log.csv", append=True, separator=';')
#checkpointer = ModelCheckpoint(filepath="resnet50_"+dataset+"_weights.{epoch:02d}-{val_accuracy:.2f}.h5", verbose=1, save_best_only=False)
original_hist2=model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=20,
    verbose=1,
    callbacks=[csv_logger],
    validation_data=validation_generator,
    validation_steps= (nb_validation_samples // batch_size))

#model.save("resnet50_"+dataset+"_end.h5")
#callbacks=[csv_logger,checkpointer],

################################################################################ 
# SAVE MODEL                                                                   #
################################################################################
model.save(final_model)


#%tensorflow_version 1.x
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from pathlib import Path
import os
from sklearn.metrics import classification_report, confusion_matrix
#import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import itertools
import pandas as pd

# dimensions des images d'entraînement
img_width, img_height = 64, 64
batch_size = 128
################################################################################
# HELPER FUNCTIONS                                                             #
################################################################################ 

def number_of_files(dirname):
	cpt = sum([len(files) for r, d, files in os.walk(dirname)])
	return cpt


################################################################################ 
# DÉFINITION DES DONNÉES D'ENTRÉE                                              #
################################################################################
#test_data_dir="EuroSatRGBmini"
nb_test_samples=number_of_files(test_data_dir)

# predict_class
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(64, 64),
		    shuffle=False,
        class_mode='categorical',
        batch_size=batch_size)

filenames = test_generator.filenames
nb_samples = len(filenames)
y_true_labels = test_generator.classes
#y_true_labels=y_true_labels.astype(int)
y_indices=test_generator.class_indices
print(test_generator.class_indices)

target_names=test_generator.class_indices.keys()
evalu = model.evaluate_generator(test_generator,steps = nb_samples // batch_size)       
print("Total samples:"+str(nb_test_samples))
print(model.metrics_names)
print(evalu)
test_generator.reset()
predicted = model.predict_generator(test_generator,steps = nb_samples / batch_size, verbose=1)
y_pred = np.rint(predicted)
predicted_class_indices=np.argmax(predicted,axis=1)

#Next step is I want the name of the classes:

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#Where by class numbers will be replaced by the class names. One final step if you want to save it to a csv file, arrange it in a dataframe with the image names appended with the class predicted.
print("=======================================================================")
print("Confusion matrix test set")
print(confusion_matrix(test_generator.classes, predicted_class_indices))
print("=======================================================================")
print("Report for test set")

#For the EuroSat network only
#target_names=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
print(classification_report(test_generator.classes, predicted_class_indices, target_names=target_names))
#cm=confusion_matrix(test_generator.class_indices, y_pred)
#print(classification_report(test_generator.class_indices, predicted_class_indices, target_names=labels))
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
filename="prediction_hyper_"+dataset+".csv"
results.to_csv(filename)
#!cp $filename /content/drive/'My Drive'/
print(results)
