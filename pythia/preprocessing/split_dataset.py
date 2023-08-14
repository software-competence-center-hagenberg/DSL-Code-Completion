import os
import shutil
import numpy as np
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="Arguments for split dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--directory", help="directory of generated files")
parser.add_argument("-i", "--input_folder", help="directory of input files")
parser.add_argument("-t", "--train_folder", help="directory of training files")
parser.add_argument("-v", "--validation_folder", help="directory of validation files")
args = parser.parse_args()
directory = args.directory
input_folder = args.input_folder
train_folder = args.train_folder
validation_folder = args.validation_folder
percentage = 0.7
files = np.array(glob(os.path.join(input_folder, '*.csv')))

if not os.path.isdir(train_folder):
    os.mkdir(train_folder)

if not os.path.isdir(validation_folder):
    os.mkdir(validation_folder)

choice = np.random.choice(np.arange(len(files)), size=np.round(len(files)*percentage).astype(np.int), replace=False)
selected = np.zeros(len(files), dtype=np.bool)
selected[choice] = True

for i,f in enumerate(files[selected]):
    shutil.copyfile(f, os.path.join(train_folder, f'{i}.csv'))

for i,f in enumerate(files[~selected]):
    shutil.copyfile(f, os.path.join(validation_folder, f'{i}.csv'))