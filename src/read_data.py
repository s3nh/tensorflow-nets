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



def data_gen(path):
    """
    Get files list
    based on path value
    """
    _files = tf.data.Dataset.list_files(os.path.join(path, '*/*'))
    return _files

def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names

# Image decoding 

def decode_img(img):
    img = tf.image.decode_jpeg(img, channel = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# Preprocess path 

def process_path(file_path):
    image_count, CLASS_NAMES = _get_data_info(file_path)
    label = get_label(file_path, CLASS_NAMES)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def get_structured_dataset(listed_files):
    labeled_ds =  listed_files.map(process_path, num_parallel_calls = AUTOTUNE)
    return labeled_ds


def main():
    files = data_gen(path = DATA_PATH)
    CLASS_NAMES, image_count = _get_data_info(DATA_PATH)
    print("\n".format([el for el in CLASS_NAMES]))
    labeled_ds = get_structured_dataset(files)
    print(labeled_ds)

if __name__ == "__main__":
    main()
