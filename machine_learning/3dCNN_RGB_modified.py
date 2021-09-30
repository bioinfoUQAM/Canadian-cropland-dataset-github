#!/usr/bin/env python
# coding: utf-8

#  # Canadian Cropland Classification Using Deep Learning : 3 Dimensional - Convolutional Neural Networks (3D-CNN)

# Reference: *Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.*
# Zunair, H., Rahman, A., Mohammed, N., & Cohen, J. P. (2020, October). Uniformizing techniques to process CT scans with 3D CNNs for tuberculosis prediction.
# In International Workshop on PRedictive Intelligence In MEdicine (pp. 156-168). Springer, Cham.

# [https://arxiv.org/abs/2007.13224]

# import libraries
import numpy as np
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, ConvLSTM2D, Conv3D, MaxPool3D, BatchNormalization, GlobalAveragePooling3D
from keras.layers.recurrent import LSTM
from keras.callbacks import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import keras.callbacks
import utils as utils
import os, random
import tensorflow as tf

tf.random.set_seed(20210928)

# define the train, validation and test directories and get the image paths
train_directory = "2019/RGB/training"
val_directory = "2019/RGB/validation"
test_directory = "2019/RGB/test"

train_imagePaths = utils.get_all_file_paths(train_directory)
val_imagePaths = utils.get_all_file_paths(val_directory)
test_imagePaths = utils.get_all_file_paths(test_directory)

# get the unique labels from the image paths
labels = utils.create_labels_set(train_directory, train_imagePaths)
print("Crop classes: ", labels)
print(" ")

# print the number of available sets per class category
for classname in labels:
    directory_name = train_directory + "/" + classname
    sequence_counts = utils.get_valid_sequence_count(utils.create_image_sets(directory_name), 3)
    print("Found %i sequences in training directory %s" % (sequence_counts, classname))

print(" ")

# transform the text labels into numerical values
lb = utils.create_binarizer(labels)

# Define the 3D-CNN model architecture
batch_size = 32
input_shape = (65, 65, 3) # image shape
n_classes = len(labels)
seq_len = 3
epochs = 15

# check the number of useable sets in each directory
train_count = utils.get_valid_sequence_count(utils.create_image_sets(train_directory), seq_len)
val_count= utils.get_valid_sequence_count(utils.create_image_sets(val_directory), seq_len)
test_count = utils.get_valid_sequence_count(utils.create_image_sets(test_directory), seq_len)

print("Number of training sets: ", train_count)
print("Number of validation sets: ", val_count)
print("Number of test sets: ", test_count)

print(" ")

# create a training and validation generator
train_gen = utils.image_batch_generator(train_directory, lb, batch_size, input_shape, n_classes)
val_gen = utils.image_batch_generator(val_directory, lb, batch_size, input_shape, n_classes)

# Build the model architecture
inputs = keras.Input(shape=(3,65,65,3))

x = Conv3D(filters=32, kernel_size=(1,2,2), activation="relu", padding="same")(inputs)
#x = MaxPool3D(pool_size=2)(x) # we do not apply MaxPooling since our initial image size is small
x = BatchNormalization()(x)

x = Conv3D(filters=32, kernel_size=(1,2,2), activation="relu")(x)
#x = MaxPool3D(pool_size=2)(x)
x = BatchNormalization()(x)

x = Conv3D(filters=32, kernel_size=(1,2,2), activation="relu")(x)
#x = MaxPool3D(pool_size=2)(x)
x = BatchNormalization()(x)

#x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#x = MaxPool3D(pool_size=2)(x)
#x = BatchNormalization()(x)

x = GlobalAveragePooling3D()(x)
x = Dense(units=512, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(n_classes, activation="sigmoid")(x)

model = Model(inputs, outputs, name="3dcnn")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model structure summary
print(model.summary())

# create a directory for storing the logs
dateTimeObj = datetime.now().strftime("%d%b%Y-%Hh%Mm%S")
img_subset = train_directory.split("/")[-2]
year = train_directory.split("/")[-3]

print("Subset name: ", img_subset)
print("Year: ", year)

# save the model
filename = "3DCNN-" + img_subset + "-" + year + "-" + str(dateTimeObj)

#CNN3D_tensor = TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1, )
csv_logger = CSVLogger(filename + "_log.csv", append=True, separator=';')
early_stopping = EarlyStopping(monitor="accuracy", patience = 5)
LR_reducer = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 10)
#model_checkpoint = ModelCheckpoint(filepath=filename+".h5", monitor="val_loss", save_best_only=True)


# fit the model to the training data
CNN3D_model = model.fit(
    train_gen,
    steps_per_epoch=train_count//batch_size,
    validation_data = val_gen,
    verbose = 1,
    callbacks = [csv_logger, early_stopping, LR_reducer],
    validation_steps=val_count//batch_size,
    epochs=epochs)

# Evaluate the model using the test set

# import some additional librairies
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
test_gen = utils.image_batch_generator(test_directory, lb, batch_size, input_shape, n_classes, mode = "eval")

# predict new labels on the test set
predIdxs = model.predict(x=test_gen, steps=(test_count//batch_size))
predIdxs = np.argmax(predIdxs, axis=1)

# get the true labels of the test set
test_set = utils.create_image_sets(test_directory)

test_labels = []
for set_name in test_set:
    n = len(set_name)
    # if the set only has 2 images or less, ignore it

    if n > 2 and n % seq_len >= 0:
        for i in range(0, n % seq_len + 1):
            # get the first n (seq_len) elements in the list
            sequence =  set_name[i:i+seq_len]

            # do a check here to see if the list is good, then return it
            if utils.is_valid_sequence(sequence, seq_len):
                # get the classname from the first image in the sequence
                label = utils.get_classname(test_directory, sequence[0])
                encoded_label = utils.encode_label(lb, (label,))
                test_labels.append(label)

test_labels = lb.transform(test_labels)


# There might be less predictions than actually present because of the division by batch size,
# so we crop the test_labels matrix so the predictions and true labels have the same size

#print("Length of predictions: ", len(predIdxs))
#print("Length of test labels: ", len(test_labels))

difference = len(test_labels)-len(predIdxs)


# print the number of available sets per class category
for classname in labels:
    directory_name = test_directory + "/" + classname
    sequence_counts = utils.get_valid_sequence_count(utils.create_image_sets(directory_name), 3)
    print("Found %i sequences in validation directory %s" % (sequence_counts, classname))

print(" ")

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(test_labels[:-difference], predIdxs,
	target_names=lb.classes_))

print(" ")
print("Confusion Matrix")
print(confusion_matrix(test_labels[:-difference], predIdxs))
