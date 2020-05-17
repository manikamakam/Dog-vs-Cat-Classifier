from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

seed(1)

val_ratio = 0.25

dataset_home='data/'
src_directory = 'data/train1/'
for file in listdir(src_directory):
    print(file)
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        copyfile(src, dst)
    elif file.startswith('dog'):

        dst = dataset_home + dst_dir + 'dogs/'  + file
        copyfile(src, dst)