# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 01:46:34 2017

@author: Anoop
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:03:05 2017

@author: Anoop
"""
from keras.preprocessing import image
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.optimizers import Adam
from keras import backend as K
import os
import cv2
import numpy as np
from PIL import Image
import pickle


def load_obj(name):
    with open('labels/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

img_width, img_height = 28, 28
NUM_CLASSES = 52
os.chdir('C:/Users/Anoop/Documents/Deeplearning_setup/projects/NIST_alphabet/')
weights_path = 'weights/CNN4layer.h5'  

#class_indices = {'7a': 51, '61': 26, '62': 27, '63': 28, '64': 29, '65': 30, '66': 31, '67': 32, '68': 33, '69': 34, '48': 7, '49': 8, '46': 5, '47': 6, '44': 3, '45': 4, '42': 1, '43': 2, '41': 0, '5a': 25, '6a': 35, '6b': 36, '6c': 37, '6d': 38, '6e': 39, '6f': 40, '77': 48, '76': 47, '75': 46, '74': 45, '73': 44, '72': 43, '71': 42, '70': 41, '79': 50, '78': 49, '59': 24, '58': 23, '55': 20, '54': 19, '57': 22, '56': 21, '51': 16, '50': 15, '53': 18, '52': 17, '4f': 14, '4d': 12, '4e': 13, '4b': 10, '4c': 11, '4a': 9})

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
    
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=input_shape))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    model.add(Conv2D(filters=32, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

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
 

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model
    
    
model = build_model()
model.load_weights(weights_path)
class_dict = load_obj('class_dict')

#img_orig = cv2.imread('D:/deeplearning_datasets/NIST/72/hsf_1/hsf_1_00000.png', 0)
#img_orig = cv2.resize(img_orig, (img_width, img_height), interpolation = cv2.INTER_AREA)
img = Image.open('D:/deeplearning_datasets/NIST/41/hsf_4/hsf_4_00000.png').convert('L')
img = img.resize((img_width, img_height), resample=Image.LANCZOS)
img = np.asarray(img)
img= img.astype('float64')/255.
img= np.expand_dims(img, axis=2)
img = np.transpose(img, (2,1,0))
img= np.expand_dims(img, axis=0)


print('before prediction')
pred = model.predict(img)


