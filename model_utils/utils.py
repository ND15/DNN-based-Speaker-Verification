from keras.layers import *
from keras.models import Sequential

from modules import DeformableConvLayer


def conv_layer(inputs, filters,
               kernel_size,
               stride,
               padding):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=stride,
               padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def basic_residual_block(inputs, filters, stride=1):
    x = inputs
    layer_0 = Sequential([
        Conv2D(filters[0],
               kernel_size=1,
               strides=stride,
               padding='valid'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters[0],
               kernel_size=3,
               strides=1,
               padding='same'),
        BatchNormalization(),
        ReLU(),

        Conv2D(filters[1],
               kernel_size=1,
               strides=1,
               padding='valid'),
        BatchNormalization(),
    ])

    x = layer_0(x)

    if stride != 1:
        skip_layer = Sequential([
            Conv2D(filters[1],
                   kernel_size=1,
                   strides=stride,
                   padding='valid'),
            BatchNormalization(),
        ])

        inputs = skip_layer(inputs)
        x = Add()([x, inputs])

    x = ReLU()(x)

    return x


def conv_residual_block(inputs, filters,
                        stride=1,
                        deformable=False):
    x = inputs

    x = conv_layer(x, filters[0], kernel_size=1,
                   stride=stride,
                   padding='valid')
    if deformable:
        x = DeformableConvLayer(filters=filters[0],
                                kernel_size=3,
                                strides=1,
                                num_deformable_group=1,
                                padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    else:
        x = conv_layer(x, filters=filters[0],
                       kernel_size=3,
                       stride=1,
                       padding='same')

    x = conv_layer(x, filters=filters[1],
                   kernel_size=1,
                   stride=1,
                   padding='same')

    if stride != 1:
        skip_layer = Sequential([
            Conv2D(filters[1],
                   kernel_size=1,
                   strides=stride,
                   padding='valid'),
            BatchNormalization(),
        ])

        inputs = skip_layer(inputs)
        x = Add()([x, inputs])

    x = ReLU()(x)

    return x
