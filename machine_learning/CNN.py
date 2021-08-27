#!/usr/bin/env python
# coding: utf-8

# # AAFC Cropland Classification Using Deep Learning

# Here, we will design an image classification pipeline for the AAFC cropland dataset. The dataset contains 6646 satellite images retrieved from Sentinel-2A using the Google Earth Engine tool. The dataset contains image patches of four types of cropland (barley, corn, pasture and mixed wood) from 5 different time periods in the year 2019 (june, july, august, september, october).

from datetime import datetime
from keras import layers
from keras import models
from keras import optimizers
import pydot
import graphviz
from keras.preprocessing.image import ImageDataGenerator # we will be using the images directly from the directories
import os


# ## Convolutional Neural Networks
#
# Convolutional neural networks take advantage of the spatial patterns organized within an image. A CNN is composed of CONV and POOl layers primarily. The CONV layers learn specific filters that help extract key features without having to design these features ourselves. The POOL layers perform dimensionality reduction on the image after every set of convolutions. In the context of image classification, our CNN may learn to detect features like edges, shapes, textures or patterns, or even higher-level features like spatial patterns in the higher layers of the network. The very last layer is a classifier that uses these higher-level features to make predictions regarding the contents of the image.
#
# In practice, CNNs give us two key benefits: local invariance and compositionality.

# #### Load the dataset

#
# # we will be working with the RGB image directory
# we will be working with the RGB image directory
train_data_dir = '../AAFC_dataset/ALL_classes/filtered/RGB/training'
validation_data_dir = '../AAFC_dataset/ALL_classes/filtered/RGB/validation'
test_data_dir = '../AAFC_dataset/ALL_classes/filtered/RGB/test'# we will be working with the GNDVI image directory
train_data_dir = '../AAFC_dataset/ALL_classes/filtered/GNDVI/training'
validation_data_dir = '../AAFC_dataset/ALL_classes/filtered/GNDVI/validation'
test_data_dir = '../AAFC_dataset/ALL_classes/filtered/GNDVI/test'# we will be working with the NDVI image directory
train_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI/training'
validation_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI/validation'
test_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI/test'# we will be working with the NDVI45 image directory
train_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI45/training'
validation_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI45/validation'
test_data_dir = '../AAFC_dataset/ALL_classes/filtered/NDVI45/test'


# we will be working with the PSRI image directory
train_data_dir = '../AAFC_dataset/ALL_classes/filtered/PSRI/training'
validation_data_dir = '../AAFC_dataset/ALL_classes/filtered/PSRI/validation'
test_data_dir = '../AAFC_dataset/ALL_classes/filtered/PSRI/test'


# create a helper function to count the number of files in each directory
def number_of_files(dirname):
    cpt = sum([len(files) for r, d, files in os.walk(dirname)])
    return cpt

nb_train_samples=number_of_files(train_data_dir)
nb_validation_samples=number_of_files(validation_data_dir)
nb_test_samples=number_of_files(test_data_dir)

print("Number of training samples: " , nb_train_samples)
print("Number of validation samples: " , nb_validation_samples)
print("Number of test samples: " , nb_test_samples)


# Define the image dimensions and other training parameters
img_width, img_height, channels  = 65, 65, 3
number_of_classes = 10
class_labels = ["Barley", "Corn", "Millet", "Mixedwood", "Oat", "Orchard", "Pasture", "Potato", "Sorghum", "Soybean"]

# training parameters
epochs = 30
batch_size = 32

steps_per_epoch = nb_train_samples//batch_size
print("Training steps per epoch: ", steps_per_epoch)
validation_steps = nb_validation_samples//batch_size
print("Validation steps per epoch: ", validation_steps)


# ### Training a simple 3 layer CONV-Net
# We will first train a simple CONV net as a baseline, and see if this model can successfully classify our four image categories.

# #### Build the model architecture

# Build the CNN
CNN_model = models.Sequential()
CNN_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)))
CNN_model.add(layers.MaxPooling2D((2, 2)))
CNN_model.add(layers.Dropout(0.25))
CNN_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN_model.add(layers.MaxPooling2D((2, 2)))
CNN_model.add(layers.Dropout(0.25))
CNN_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN_model.add(layers.MaxPooling2D((2, 2)))
CNN_model.add(layers.Flatten())
CNN_model.add(layers.Dropout(0.5))
CNN_model.add(layers.Dense(128, activation ="relu"))
CNN_model.add(layers.Dense(10, activation="softmax"))
CNN_model.summary()


# compile the model
CNN_model .compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
             metrics =["accuracy"])

# create the generators for the training, validation and test sets
train_datagen = ImageDataGenerator(rescale = 1./255) # Normalize the images
test_datagen = ImageDataGenerator(rescale = 1./255)

# training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle = True,
    class_mode='categorical')

# validation
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    shuffle = False,
    batch_size=batch_size,
    class_mode='categorical')

# test
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle = False,
    class_mode='categorical')

# Train the model
CNN_history = CNN_model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps)


dateTimeObj = datetime.now().strftime("%d%b%Y-%Hh%Mm%S")
#print(dateTimeObj)

subset = train_data_dir.split("/")[-2]
print("Subset name: ", subset)

# save the model
CNN_model.save("CNN-AAFC-" + subset + "-" + str(dateTimeObj) + ".h5")


# Plot the training results
import matplotlib.pyplot as plt

# Visualize the and accuracy loss during training
history_dict = CNN_history.history

training_loss = history_dict["loss"]
training_accuracy = history_dict["accuracy"]

validation_accuracy = history_dict["val_accuracy"]
validation_loss = history_dict["val_loss"]

epochs = range(1, len(training_loss)+1)

# Plot the train/val loss
plt.plot(epochs, training_loss, 'bo', label="Training Loss")
plt.plot(epochs, validation_loss, 'r', label=" Validation Loss")

plt.title(subset)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()

# Plot the train/val accuracy
plt.plot(epochs, training_accuracy, 'bo', label="Training Accuracy")
plt.plot(epochs, validation_accuracy, 'r', label=" Validation Accuracy")

plt.title(subset)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


from keras.utils import plot_model
plot_model(CNN_model,
    to_file="CNN_model3.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

#Confution Matrix and Classification Report
Y_pred = CNN_model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print(" ")

print('Classification Report')
print(" ")
target_names = class_labels
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
print(" ")



#Confution Matrix and Classification Report
print("TEST DATA CONFUSION MATRIX")
Y_pred = CNN_model.predict_generator(test_generator, nb_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print(" ")

print('Classification Report')
print(" ")
target_names = class_labels
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
print(" ")
