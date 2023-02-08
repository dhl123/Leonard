keras_file="./lite.h5"
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import models
import operator
from collections import defaultdict
import pickle
import sys
import argparse
from time import *
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model',
                    help='param file file')
args = parser.parse_args()
#converter.experimental_new_quantizer = True
#from tensorflow import keras
#from tensorflow import lite
def loss_fn(y_true, y_pred):
        return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)
#model = keras.models.load_model(keras_file)
model = getattr(models, 'LSTM_multi')(2048, 10, 25)
model.load_weights(args.model)
#model = keras.models.load_model(keras_file,custom_objects={"loss_fn": loss_fn})
#begin=time()
#converter.experimental_new_quantizer = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter=tf.lite.TFLiteConverter.from_keras_model(keras_file)
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
#converter = lite.TFLiteConverter.from_keras_model_file(keras_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
#end=time()
#print(end-begin)
open(keras_file+'2048'+args.model, "wb").write(tflite_model)
