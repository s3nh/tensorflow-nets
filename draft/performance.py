import tensorflow as tf 

"""
Some Pipeline Optimization performance 
"""
# Create synthetic dataset, 
# Which will simulate opening/reading file 

class SyntheticData(tf.data.Dataset):

    def _generator(num_samples):
        """
        num_samples:
            number of observations 
        """
        # Fall asleep for a while
        time.sleep(0.03)
        for sample_idx in range(num_samples):
            # Reading data
            time.sleep(0.015)
            yield(sample_idx, )

    # Define new object
   def __new__(cls, num_samples = 3):
       return tf.data.Dataset.from_generator(
           cls._generator, 
           output_types = tf.dtypes.int64, 
           output_shapes = (1,), 
           args = (num_samples, )
       ) 