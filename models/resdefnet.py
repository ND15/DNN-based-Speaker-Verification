import numpy
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential, Model
from model_utils.modules import DeformableConvLayer
from model_utils.modules import TimeFrequencyAttentionModule

"""
    Paper Title: Speaker Verification System Based on Deformable 
                 CNN and Time-Frequency Attention
    
    Paper Link: http://www.apsipa.org/proceedings/2020/pdfs/0001689.pdf
    
    This is model is based on image-like representation of 
"""


class ResidualBlock(Layer):
    def __init__(self, filters, strides, padding, deformable, attention=False,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.deformable = deformable
        self.attention = attention

        self.conv_layers_0 = [
            Conv2D(filters=self.filters[0],
                   kernel_size=1,
                   strides=self.strides,
                   padding="valid"),
            BatchNormalization(),
            ReLU()
        ]

        self.deformable_layers = [
            DeformableConvLayer(filters=self.filters[0],
                                kernel_size=3,
                                strides=1,
                                num_deformable_group=1,
                                padding='same'),
            BatchNormalization(),
            ReLU()
        ]

        self.conv_layers_1 = [
            Conv2D(filters=self.filters[0],
                   kernel_size=3,
                   strides=1,
                   padding="same"),
            BatchNormalization(),
            ReLU()
        ]

        self.conv_layers_2 = [
            Conv2D(filters=self.filters[1],
                   kernel_size=1,
                   strides=1,
                   padding="same"),
            BatchNormalization(),
            ReLU()
        ]

        self.conv_skip = Conv2D(self.filters[1],
                                kernel_size=1,
                                strides=self.strides,
                                padding='valid')
        self.batch_skip = BatchNormalization()
        self.add = Add()
        self.relu_out = ReLU()
        self.time_frequency_module = TimeFrequencyAttentionModule()

    def call(self, inputs, *args, **kwargs):
        x = inputs

        for layer in self.conv_layers_0:
            x = layer(x)

        if self.deformable:
            for layer in self.deformable_layers:
                x = layer(x)
        else:
            for layer in self.conv_layers_1:
                x = layer(x)

        for layer in self.conv_layers_2:
            x = layer(x)

        if self.strides != 1:
            inputs = self.conv_skip(inputs)
            inputs = self.batch_skip(inputs)

            x = self.add([x, inputs])

        x = self.relu_out(x)

        if self.attention:
            x = self.time_frequency_module(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {"filters": self.filters,
             "padding": self.padding,
             "strides": self.strides,
             "deformable": self.deformable}
        )
        return config


class Deformable_CNN(keras.Model):
    def __init__(self, n_classes=1, **kwargs):
        super(Deformable_CNN, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.conv_layer_0 = Conv2D(filters=64,
                                   kernel_size=7,
                                   strides=1,
                                   padding='SAME')

        self.max_pool_0 = MaxPool2D(pool_size=(2, 2),
                                    strides=(2, 2))

        self.conv_res_block_0_0 = ResidualBlock((48, 96), strides=1,
                                                padding='valid',
                                                deformable=True)

        self.conv_res_block_0_1 = ResidualBlock((48, 96), strides=1,
                                                padding='valid',
                                                deformable=False,
                                                attention=True)

        self.conv_res_block_1_0 = ResidualBlock((96, 128), strides=1,
                                                padding='valid',
                                                deformable=True)

        self.conv_res_block_1_1 = ResidualBlock((96, 128), strides=1,
                                                padding='valid',
                                                deformable=False)

        self.conv_res_block_1_2 = ResidualBlock((96, 128), strides=2,
                                                padding='valid',
                                                deformable=False,
                                                attention=True)

        self.conv_res_block_2_0 = ResidualBlock((128, 256), strides=1,
                                                padding='valid',
                                                deformable=True)

        self.conv_res_block_2_1 = ResidualBlock((128, 256), strides=1,
                                                padding='valid',
                                                deformable=False)

        self.conv_res_block_2_2 = ResidualBlock((128, 256), strides=2,
                                                padding='valid',
                                                deformable=False,
                                                attention=True)

        self.conv_res_block_3_0 = ResidualBlock((256, 512), strides=1,
                                                padding='valid',
                                                deformable=True)

        self.conv_res_block_3_1 = ResidualBlock((256, 512), strides=1,
                                                padding='valid',
                                                deformable=False)

        self.conv_res_block_3_2 = ResidualBlock((256, 512), strides=2,
                                                padding='valid',
                                                deformable=False,
                                                attention=True)

        self.max_pool_1 = MaxPool2D(pool_size=(3, 1),
                                    strides=(2, 2))

        self.conv_layer_1 = Conv2D(filters=512, kernel_size=(7, 1),
                                   strides=(1, 1))

        self.flatten = Flatten()

        self.dense_0 = Dense(units=512)

        self.outputs = Dense(units=self.n_classes, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=True, mask=None):
        x = self.conv_layer_0(inputs)
        x = self.max_pool_0(x)

        x = self.conv_res_block_0_0(x)
        x = self.conv_res_block_0_1(x)

        x = self.conv_res_block_1_0(x)
        x = self.conv_res_block_1_1(x)
        x = self.conv_res_block_1_2(x)

        x = self.conv_res_block_2_0(x)
        x = self.conv_res_block_2_1(x)
        x = self.conv_res_block_2_2(x)

        x = self.conv_res_block_3_0(x)
        x = self.conv_res_block_3_1(x)
        x = self.conv_res_block_3_2(x)

        x = self.max_pool_1(x)
        x = self.conv_layer_1(x)
        x = self.flatten(x)
        x = self.dense_0(x)

        return self.outputs(x)


if __name__ == "__main__":
    model = Deformable_CNN(n_classes=1)
    p = tf.random.normal((10, 256, 128, 1))
    y = numpy.random.randint(low=0, high=1, size=(10, 1), dtype=int)
    y = tf.cast(y, tf.int32)
    print(y.shape)

    model.compile(loss=keras.losses.binary_crossentropy)
    model.fit(p, y, batch_size=2)
