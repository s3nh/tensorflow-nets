import tensorflow as tf 
keras = tf.keras
import os 
import pathlib 
import matplotlib.pyplot as plt 
import numpy as np 
from  src.utils import _get_data_info
# Set precision 

DATA_PATH = 'data/'
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES, image_count = _get_data_info(DATA_PATH)
IMG_WIDTH, IMG_HEIGHT = 224, 224

print("Number of images to preprocess {}".format(image_count))


def data_gen(path):
    """
    Get files list
    based on path value
    """
    path =  pathlib.Path(path) 
    _files = tf.data.Dataset.list_files(str(path/'*/*'))
    return _files

def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names

# Image decoding 

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# Preprocess path 

def process_path(file_path):
    label = get_label(file_path, CLASS_NAMES)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def get_structured_dataset(listed_files):
    labeled_ds =  listed_files.map(process_path, num_parallel_calls = AUTOTUNE)
    return labeled_ds


def main():
    files = data_gen(path = DATA_PATH)
    print(CLASS_NAMES)
    labeled_ds = get_structured_dataset(files)
    print(labeled_ds)

if __name__ == "__main__":
    main()
