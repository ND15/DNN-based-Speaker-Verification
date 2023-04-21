import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, PReLU, Dense, Input, Flatten
from keras.layers import LeakyReLU, ReLU, GlobalAveragePooling1D, Layer, Multiply, LayerNormalization
from keras.models import Model, Sequential
import keras.backend as k

"""
    Paper Name: Time-domain Speaker Verification Using
                Temporal Convolutional Networks

    Paper Link: https://ieeexplore.ieee.org/document/9414765
"""


# Custom SENet Layer
class SENet(Layer):
    def __init__(self, out_dim, ratio, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.ratio = ratio
        self.se = Sequential([
            GlobalAveragePooling1D(),
            Dense(units=(self.out_dim / self.ratio)),
            ReLU(),
            Dense(units=self.out_dim, activation='sigmoid'),
        ])

        self.multiply = Multiply()

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.se(x)
        # print(x.shape)
        x = self.multiply([x, inputs])
        return x


def senet_block(inputs, ratio=4):
    b, l, c = inputs.shape
    x = GlobalAveragePooling1D()(inputs)
    x = Dense(units=c // ratio)(x)
    x = ReLU()(x)
    x = Dense(units=c, activation='sigmoid')(x)
    x = Multiply()([x, inputs])
    return x


def _conv_block(inputs, filters, kernel_size=3):
    x = Conv1D(filters=filters, kernel_size=kernel_size)(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def _d_conv_block(inputs, filters, kernel_size=3, dilation_factor=1):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal',
               dilation_rate=dilation_factor)(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def conv_se_block(inputs, ratio=8, conv_filters=32, d_conv_filters=32,
                  kernel_size=3, dilation_factor=1):
    x = _conv_block(inputs, filters=conv_filters, kernel_size=kernel_size)
    x = _d_conv_block(x, filters=d_conv_filters, kernel_size=kernel_size,
                      dilation_factor=dilation_factor)
    x = senet_block(x, ratio=ratio)
    return x


def conv_se_model(inputs=Input((32000, 1)), stack_length=4, num_speakers=24):
    """
    TODO:
        1. SEResNet
        2. HalfResNet
    """
    x = _conv_block(inputs, filters=16)

    for i in range(stack_length):
        x = _d_conv_block(x, filters=32 * (i + 1), kernel_size=5, dilation_factor=2 ** i)

    x = MaxPool1D(3)(x)

    for i in range(stack_length):
        x = _d_conv_block(x, filters=32 * (i + 1), kernel_size=5, dilation_factor=2 ** i)

    x = MaxPool1D(3)(x)

    for i in range(stack_length):
        x = _d_conv_block(x, filters=32 * (i + 1), kernel_size=5, dilation_factor=2 ** i)

    x = MaxPool1D(3)(x)

    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation='relu')(x)  # embeddings

    x = Dense(num_speakers, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
