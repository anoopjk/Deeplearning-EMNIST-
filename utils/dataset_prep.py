# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:50:50 2017

@author: Anoop
"""

import os
from distutils.dir_util import copy_tree

#training set
src_root_dir = "D:/deeplearning_datasets/NIST/"
src_root_folders = os.listdir(src_root_dir)
dst_root_dir = "D:/deeplearning_datasets/NIST_alphabet/"

dst_root_folders = os.listdir(src_root_dir)
for i in dst_root_folders:
    directory = os.path.join(dst_root_dir, i)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
for i in src_root_folders:
    print("folder: ", i)
    src_subfolders = os.listdir(os.path.join(src_root_dir, i))
    for j in src_subfolders:
        if ( j == "train_" + i):
            #shutil.copy(os.path.join(src_root_dir, i, j), os.path.join(dst_root_dir, i))
            copy_tree(os.path.join(src_root_dir, i, j), os.path.join(dst_root_dir, i))
            
################################################################################################################
#  validation set

src_root_dir = "D:/deeplearning_datasets/NIST/"
src_root_folders = os.listdir(src_root_dir)
dst_root_dir = "D:/deeplearning_datasets/NIST_alphabet/validation/"

ignore = [str(i) for i in range(30,40)]

dst_root_folders = os.listdir(src_root_dir)
for i in dst_root_folders:
    if i not in ignore:
        directory = os.path.join(dst_root_dir, i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
for i in src_root_folders:
    if i not in ignore:
        print("folder: ", i)
        src_subfolders = os.listdir(os.path.join(src_root_dir, i))
        for j in src_subfolders:
            if ( j == "hsf_4"):
                #shutil.copy(os.path.join(src_root_dir, i, j), os.path.join(dst_root_dir, i))
                copy_tree(os.path.join(src_root_dir, i, j), os.path.join(dst_root_dir, i))
            
            
            

