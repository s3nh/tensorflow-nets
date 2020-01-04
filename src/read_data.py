import tensorflow as tf 
keras = tf.keras
import os 
import pathlib 
import matplotlib.pyplot as plt 
import numpy as np 

# Set precision 


DATA_PATH = 'data'
AUTOTUNE = tf.data.experimental.AUTOTUNE


def data_gen(path):
    """
    Get files list
    based on path value
    """
    _files = tf.data.Dataset.list_files(os.path.join(path, '*/*'))
    return _files

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES

# Image decoding 

def decode_img(img):
    img = tf.image.decode_jpeg(img, channel = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

# Preprocess path 

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_structured_dataset(file_path):
    labeled_ds =  file_path.map(process_path, num_parallel_calls = AUTOTUNE)
    return labeled_ds




def main():
    files = data_gen(path = DATA_PATH)
    for file in files.take(5):
        print(file.numpy())
    labels = get_label(files)
    print(labels)


if __name__ == "__main__":
    main()
