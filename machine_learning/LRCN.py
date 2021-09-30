
#!/usr/bin/env python
# coding: utf-8

#  # AAFC Canadian Cropland Classification Using Deep Learning : Long-Term Recurrent Convolutional Network (LRCN)


# Reference: *Long-term recurrent convolutional networks for visual recognition and description.*
# J. Donahue, L. Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, T. Darrell, and K. Saenko.
# CVPR , page 2625-2634. IEEE Computer Society, (2015)
# 
# [http://cs231n.stanford.edu/reports2016/221_Report.pdf]
# [https://arxiv.org/pdf/1411.4389v3.pdf]


# import libraries 
import numpy as np
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, ConvLSTM2D
from keras.layers.recurrent import LSTM
from keras.callbacks import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import keras.callbacks
import AAFC_utils as utils 
import os, random


# define the train, validation and test directories and get the image paths 
train_directory = "../../AAFC-dataset/2019/RGB/training"
val_directory = "../../AAFC-dataset/2019/RGB/validation"
test_directory = "../../AAFC-dataset/2019/RGB/test"

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


# Define the LRCN model architecture
batch_size = 32
input_shape = (65, 65, 3) # image shape 
n_classes = len(labels)
seq_len = 3
epochs = 30

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

# use simple CNN structure as convolutional base  
model = Sequential()

model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=(seq_len,)+input_shape))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Dropout(0.5))

model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Dropout(0.5))

#model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
#model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size=(1, 2, 2)))
#model.add(Dropout(0.5))

model.add(Dense(320))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# reshape before passing to LSTM
out_shape = model.output_shape
print('====Model output shape: ', out_shape)
model.add(Reshape((seq_len, out_shape[2] * out_shape[3] * out_shape[4])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
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
filename = "LRCN-AAFC-" + img_subset + "-" + year + "-" + str(dateTimeObj)

LCRN_tensor = TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1, )
csv_logger = CSVLogger("logs/" + filename + "_log.csv", append=True, separator=';')
early_stopping = EarlyStopping(monitor="accuracy", patience = 5)
LR_reducer = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 3) #min_lr=0.001
model_checkpoint = ModelCheckpoint(filepath=filename+".h5", monitor="val_loss", save_best_only=True) 


# fit the model to the training data 
LSTM_model = model.fit(
    train_gen,
    steps_per_epoch=train_count//batch_size,
    validation_data = val_gen,
    verbose = 1, 
    callbacks = [csv_logger, early_stopping, LR_reducer, model_checkpoint],
    validation_steps=val_count//batch_size, 
    epochs=epochs)


# Plot the training results
import matplotlib.pyplot as plt 

# Visualize the and accuracy loss during training 
history_dict = LSTM_model.history

training_loss = history_dict["loss"]
training_accuracy = history_dict["accuracy"]

validation_accuracy = history_dict["val_accuracy"]
validation_loss = history_dict["val_loss"]

epochs = range(1, len(training_loss)+1) 

# Plot the train/val accuracy
plt.plot(epochs, training_accuracy, 'bo', label="Training Accuracy")
plt.plot(epochs, validation_accuracy, 'r', label=" Validation Accuracy")

plt.title("LRCN training accuracy: " + year + "/" + img_subset)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
#plt.show()
plt.savefig(filename+"_accuracy", dpi=300)

plt.figure()

# Plot the train/val loss
plt.plot(epochs, training_loss, 'bo', label="Training Loss")
plt.plot(epochs, validation_loss, 'r', label=" Validation Loss")

plt.title("LCRN training loss: " + year + "/" + img_subset)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(filename+"_loss", dpi=300)
#plt.show()


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

# get the true labels of the test 
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
                label = utils.get_classname(test_directory, sequence[0])
                encoded_label = utils.encode_label(lb, (label,))
                #print("Label : ", (label, encoded_label))
                test_labels.append(label)
 
test_labels = lb.transform(test_labels)


# There might be less predictions than actually present because of the division by batch size, 
# so we crop the test_labels matrix so the predictions and true labels have the same size

print("Length of predictions: ", len(predIdxs))
print("Length of test labels: ", len(test_labels))

print(predIdxs)
print(test_labels)

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


