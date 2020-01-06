import tensorflow as tf 

def _classification_head():
   global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
   return global_average_layer

def _prediction_layer():
   feature_batch_average = _classification_head()
   prediction_layer = tf.keras.layers.Dense(1)
   #prediction_batch = prediction_layer(feature_batch_average)
   return prediction_layer

def load_model(IMG_SHAPE= (224, 224, 3)):
   base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, 
   include_top = False, 
   weights = 'imagenet') 
   base_model.trainable = False 
   global_average_layer = _classification_head()
   prediction_layer = _prediction_layer()
   model = tf.keras.Sequential([
      base_model, 
      global_average_layer, 
      prediction_layer
   ])
   return model

def main():
   load_model()
if __name__ == "__main__":
    main()