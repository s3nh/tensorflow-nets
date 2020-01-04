import tensorflow as tf 
keras = tf.keras

import pathlib 
import matplotlib.pyplot as plt 
import numpy as np 

# Set precision 


DATA_PATH = 'data/food-101'



def data_gen(path):
    _files = tf.data.Dataset.list_files(path/'*/*')
    return _files

def main():

    files = data_gen(path = DATA_PATH)

if __name__ == "__main__":
    main()
