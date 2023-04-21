import tensorflow as tf
from tensorflow import keras
from models.temporal_cnn import SENet

if __name__ == "__main__":
    x = tf.random.normal((10, 128, 1))
    x = keras.layers.Conv1D(32, kernel_size=3)(x)
    se = SENet(32, ratio=4)(x)
    print(se.shape)
