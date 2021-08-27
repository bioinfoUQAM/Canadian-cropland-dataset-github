from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
from pathlib import Path
import os


# dimensions des images d'entraînement
img_width, img_height = 64, 64

###############################################################################
# HELPER FUNCTIONS
def number_of_files(dirname):
	cpt = sum([len(files) for r, d, files in os.walk(dirname)])
	return cpt

################################################################################ 
# DÉFINITION DES DONNÉES D'ENTRÉE                                              #
################################################################################
train_data_dir = '../data/RGB/training'
validation_data_dir = '../data/RGB/validation'
test_data_dir = '../data/RGB/test'
nb_train_samples=number_of_files(train_data_dir)
nb_validation_samples=number_of_files(validation_data_dir)

epochs_pre = 10     # Nombre d'itérations
epochs_last = 20     # Nombre d'itérations
batch_size = 32 # Nombre d'images à la fois


base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
################################################################################ 
# DÉFINITION DE L'AUGMENTATION DES IMAGES                                      #
################################################################################

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator() # On normalise les images

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
	shuffle = True,
    class_mode='categorical')

#train_datagen.fit(train_generator)	
	
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

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=5, min_lr=0.001)

tensor=TensorBoard(log_dir='logs',histogram_freq=1,embeddings_freq=1,)
csv_logger = CSVLogger('ResNet50_20210501_log.csv', append=True, separator=';')

#fit_generator(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
original_hist=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs_pre,
    verbose=1,
    callbacks=[csv_logger],
    validation_data=validation_generator,
    validation_steps= (nb_validation_samples // batch_size))

model.save_weights('resnet50_rgb_first.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers:
   layer.trainable = True
   #174
#for layer in model.layers[:142]:
#   layer.trainable = False
#for layer in model.layers[142:]:
#   layer.trainable = True
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
csv_logger = CSVLogger('resnet50_rgb_last_log.csv', append=True, separator=';')
checkpointer = ModelCheckpoint(filepath='/resnet50_rgb_weights.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
original_hist2=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs_last,
    verbose=1,
    callbacks=[csv_logger,checkpointer],
    validation_data=validation_generator,
    validation_steps= (nb_validation_samples // batch_size))
model.save("model.resnet50_rgb_end.h5")
model.save_weights('weights.resnet50_rgb_end.h5')

# ################################################################################ 
# # VISUALISATION DES RÉSULTATS D'ENTRAÎNEMENT                                   #
# ################################################################################

# import matplotlib.pyplot as plt
# original_hist.history

# acc = original_hist.history['acc']
# val_acc = original_hist.history['val_acc']
# loss = original_hist.history['loss']
# val_loss = original_hist.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss ')
# plt.legend()

# plt.show()

# plt.clf()   # Création d'une nouvelle figure
# acc_values = original_hist.history['acc']
# val_acc_values = original_hist.history['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()