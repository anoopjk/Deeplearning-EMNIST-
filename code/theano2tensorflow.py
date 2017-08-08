import os
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.optimizers import Adam
from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

img_width, img_height = 28, 28
MODEL_NAME = 'nist_convnet'
NUM_CLASSES = 52

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=input_shape))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    model.add(Conv2D(filters=32, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    model.add(Conv2D(filters=64, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, \
            padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
 

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model
    


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    model = build_model()
    
    os.chdir('C:/Users/Anoop/Documents/Deeplearning_setup/projects/NIST_alphabet/')
    
    if not os.path.exists('out'):
        os.mkdir('out')
    model.load_weights('weights/CNN5layer.h5')
    ops = []
    for layer in model.layers:
       if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
          print(layer.__class__.__name__)
          original_w = K.get_value(layer.W)
          converted_w = convert_kernel(original_w)
          ops.append(tf.assign(layer.W, converted_w).op)
              
    K.get_session().run(ops)
    model.save_weights('CNN5layer_tensorflow.h5')

    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()
