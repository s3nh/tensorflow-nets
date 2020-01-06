import tensorflow as tf 
from tensorflow.keras  import backend as K 

def _get_layer_output(x, model = keras.Model() , n_layer = 3, output = False):
    _output = K.function([model.layers[0].input], 
                        [model.layers[n_layers].output])
    layer_output = _output([x])[0]
    if output:
        return layer_output, _output 
    else:
        return layer_output 



def _get_layer_info()
def main():
    _get_layer_output()


if __name__ == "__main__":
    main()