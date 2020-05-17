from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

seed(1)

val_ratio = 0.25

dataset_home='data/'
src_directory = 'data/train1/'

if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/train/cats/"):
    os.makedirs("data/train/cats/")
if not os.path.exists("data/train/dogs/"):
    os.makedirs("data/train/dogs/")

if not os.path.exists("data/val"):
    os.makedirs("data/val")
if not os.path.exists("data/val/cats/"):
    os.makedirs("data/val/cats/")
if not os.path.exists("data/val/dogs/"):
    os.makedirs("data/val/dogs/")
for file in listdir(src_directory):
    print(file)
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'val/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        copyfile(src, dst)
    elif file.startswith('dog'):

        dst = dataset_home + dst_dir + 'dogs/'  + file
        copyfile(src, dst)