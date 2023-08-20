import math

import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers
from keras.layers import Dropout, Dense
from keras.models import Model
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import *


def create_sp_classifier():
    input = keras.Input(shape=(180, 2))
    input_with_adjacent_segment = keras.Input(shape=(900, 2))

    feature = keras.layers.Conv1D(filters=16, kernel_size=11, strides=1, padding='same', activation='relu')(input)
    feature_flatten = layers.Flatten()(feature)
    feature_flatten = layers.Dense(1024, activation="relu")(feature_flatten)
    dp = Dropout(0.8)(feature_flatten)

    outputs = Dense(2, activation='softmax', name="Output_Layer")(dp)
    model = Model(inputs=[input, input_with_adjacent_segment], outputs=outputs)
    return model
