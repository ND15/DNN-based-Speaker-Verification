import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Model, Sequential

conv_filters = [80, 60, 60]
conv_kernel_size = [255, 5, 5]
dense_units = [2048, 2048, 1024]


def block(inputs, index, b_type="conv",
          norm_type='batch'):
    if b_type == "conv":
        x = Conv1D(filters=conv_filters[index], kernel_size=conv_kernel_size[index],
                   strides=2, padding='valid',
                   name=f'{b_type}_layer_{index}')(inputs)
    else:
        x = Dense(units=dense_units[index], name=f'{b_type}_fc_layer_{index}')(inputs)

    if b_type == "conv":
        x = MaxPooling1D(pool_size=3, name=f'{b_type}_pool_layer_{index}')(x)

    if norm_type == 'layer':
        x = LayerNormalization(name=f'{b_type}_{norm_type}_layer_{index}')(x)

    else:
        x = BatchNormalization(momentum=0.05, name=f'{b_type}_{norm_type}_layer_{index}')(x)

    x = LeakyReLU(alpha=0.2, name=f'{b_type}_l_relu_{index}')(x)

    return x


def model(inputs=Input((32000, 1))):
    x = block(inputs, index=0, b_type="conv", norm_type='layer')
    # x = Dropout(0.2)(x)
    x = block(x, index=1)
    # x = Dropout(0.2)(x)
    x = block(x, index=2)
    x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = block(x, index=0, b_type="dense")
    # x = Dropout(0.2)(x)
    x = block(x, index=1, b_type="dense")
    # x = Dropout(0.2)(x)
    x = block(x, index=2, b_type="dense")

    outputs = Dense(units=24, activation='softmax', name='output_layer')(x)

    base_model = Model(inputs=inputs, outputs=outputs, name="Baseline_model")

    return base_model


if __name__ == '__main__':
    m = model()
    m.summary()
