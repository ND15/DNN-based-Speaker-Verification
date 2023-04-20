import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model
from abc import ABC

"""
    Paper Name: RawNet: Advanced end-to-end deep neural network using raw waveforms
                for text-independent speaker verification
    
    Paper Link: https://arxiv.org/pdf/1904.08104.pdf
"""

BATCH = 64


class ResidualLayer(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.block = [
            Conv1D(filters=filters[0],
                   kernel_size=kernel_size,
                   strides=1, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            Conv1D(filters=filters[1],
                   kernel_size=3,
                   strides=1, padding="same"),
            BatchNormalization(),
        ]

        self.skip_block = [
            Conv1D(filters=filters[1],
                   kernel_size=1,
                   strides=1,
                   padding='same'),
            BatchNormalization()
        ]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.block:
            x = layer(x)

        if K.int_shape(inputs)[-1] != K.int_shape(x)[-1]:
            for layer in self.skip_block:
                inputs = layer(inputs)

        x = Add()([inputs, x])
        x = LeakyReLU(0.2)(x)

        return x


class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.centers = None
        self.result = None

    def build(self, input_shape):
        self.centers = self.add_weight(name="centers",
                                       shape=(256, 24),
                                       initializer="uniform",
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None, **kwargs):
        # print(tf.transpose(x[1]).shape)
        # print(self.centers.shape)
        delta_centers = K.dot(K.transpose(x[1]), K.dot(x[1], self.centers) - x[0])
        centers_count = K.sum(K.transpose(x[1]), axis=-1, keepdims=True) + 1
        delta_centers /= centers_count
        new_centers = self.centers - self.alpha * delta_centers

        self.add_update([(self.centers, new_centers), x])

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class RawNet_Module(Model, ABC):
    def __init__(self, res_blocks=4, dense_block=3, feature_dim=1024,
                 **kwargs):
        super(RawNet_Module, self).__init__(**kwargs)
        self.res_blocks = res_blocks
        self.dense_blocks = dense_block
        self.feature_dim = feature_dim

        self.conv_1 = Conv1D(128, 3, 3, padding='valid')
        self.batch_1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(0.1)

        self.residua_1 = ResidualLayer([128, 128], kernel_size=3)
        self.residua_2 = ResidualLayer([32, 32], kernel_size=3)
        # self.residua_3 = ResidualLayer([256, 256], kernel_size=3)
        # self.residua_4 = ResidualLayer([128, 128], kernel_size=3)
        # self.residua_5 = ResidualLayer([64, 64], kernel_size=3)

        self.flatten = Flatten()
        self.dense_1 = Dense(256)
        self.batch_2 = BatchNormalization()
        self.leaky_2 = LeakyReLU(0.1)

        self.output_labels = Dense(24, activation='softmax')

    def call(self, inputs, *args, **kwargs):
        x = inputs

        x = Reshape((32000, 1))(x)

        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.leaky_1(x)

        x = self.residua_1(x)
        x = MaxPool1D(pool_size=3)(x)
        x = self.residua_2(x)
        x = MaxPool1D(pool_size=3)(x)
        # x = self.residua_3(x)
        # x = MaxPool1D(pool_size=3)(x)
        # x = self.residua_4(x)
        # x = MaxPool1D(pool_size=3)(x)
        # x = self.residua_5(x)
        # x = MaxPool1D(pool_size=3)(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.batch_2(x)
        x = self.leaky_2(x)

        features = tf.identity(x)

        x = self.output_labels(x)

        return x, features


class RawNet(Model, ABC):
    def __init__(self, model, **kwargs):
        super(RawNet, self).__init__(kwargs)
        self.model = model
        self.center_loss = CenterLossLayer()

    def train_step(self, data):
        X_train, y_train = data

        with tf.GradientTape() as tape:
            y_pred, features = self.model(X_train, training=True)
            loss = self.compiled_loss(y_train, y_pred)
            c_loss = self.center_loss([y_pred, features])
            loss = loss + c_loss

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        self.compiled_metrics.update_state(y_train, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred, features = self.model(x, training=False)
            loss = self.compiled_loss(y, y_pred)
            c_loss = self.center_loss([y_pred, features])
            loss = loss + c_loss

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    module = RawNet_Module()
    module.build((128, 32000, 1))
    module.summary()
    m = RawNet(module)
    m.compile(loss='categorical_crossentropy', metrics=['accuracy'])
