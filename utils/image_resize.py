# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:51:12 2017

@author: Anoop
"""

import os
from PIL import Image


root_dir = "D:/deeplearning_datasets/NIST_alphabet/"
root_folders = os.listdir(root_dir)
file_limit = 5000

for i in root_folders:
    print("folder: ", i)
    directory = os.path.join(root_dir, i)
    img_files = os.listdir(directory)
    if  len(img_files) > file_limit:
        extra_files = len(img_files) - file_limit
        for i in range(extra_files):
            os.remove(os.path.join(directory, img_files[file_limit+i]))

#    for j, imgname in enumerate(img_files):
#        img = Image.open(os.path.join(directory, imgname))
#        img = img.resize((28, 28), Image.LANCZOS)
