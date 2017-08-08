# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:17:28 2017

@author: Anoop
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:27:45 2017

@author: Anoop
"""


import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.optimizers import Adam
from keras import backend as K
import pickle
import tensorflow as tf
#from IPython.display import display
#from PIL import Image

def save_obj(obj, name ):
    with open('labels/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



NUM_CLASSES = 52
nb_train_samples = 205561
nb_validation_samples = 23670


# dimensions of our images.
img_width, img_height = 28, 28
os.chdir('C:/Users/Anoop/Documents/Deeplearning_setup/projects/NIST_alphabet/')
train_data_dir = 'D:/deeplearning_datasets/NIST_alphabet/train/'
validation_data_dir = 'D:/deeplearning_datasets/NIST_alphabet/validation/'
weights_path = 'weights/CNN7layer.h5'    


epochs = 30
batch_size = 256

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=input_shape))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    model.add(Conv2D(filters=64, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    model.add(Conv2D(filters=128, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, \
            padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
 

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255)
    


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode= 'grayscale',
    batch_size=batch_size,
    class_mode='categorical')

print('class_indices', train_generator.class_indices)
 

save_obj(train_generator.class_indices, 'class_dict')


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode= 'grayscale',
    batch_size=batch_size,
    class_mode='categorical')
    
model = build_model() 

model.compile(loss=keras.losses.categorical_crossentropy, \
              optimizer=keras.optimizers.Adadelta(), \
              metrics=['accuracy'])
              
print("training............")
              
model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch = nb_train_samples // batch_size,
    verbose=1, 
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
    

print("saving..........")
model.save_weights(weights_path)




