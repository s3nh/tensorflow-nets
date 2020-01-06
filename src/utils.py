import numpy as np 
import tensorflow as tf
import pathlib
import yaml 
"""
Utility functions 
preparation for training 
"""

def read_config(path = 'config/config.yaml'):
    """
    Args:
        path: 
            path to config.yaml file
    return:
        config
    """

    with open(path, 'rb') as confile:
        config = yaml.safe_load(confile)
    return config

def _get_data_info(data_dir, ff = '.jpg'):
    """
    Return information about 
    data and its labels 
    Args:
        data_dir:
            data_dir to preprocess 
        ff: 
            format of preprocess files 
    """
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*{}'.format(ff))))
    assert image_count > 0, 'Indicated path is empty'
    CLASS_NAMES =  np.array([item.name for item in data_dir.glob('*')])
    return CLASS_NAMES, image_count 


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    """
    Args:
        ds: 

        cache:

        shuffle_buffer_size:
    return:
        ds:
            shuffled/batched dataset
    """
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size = shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

