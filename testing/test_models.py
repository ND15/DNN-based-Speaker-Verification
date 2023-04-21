import tensorflow as tf
from tensorflow import keras
from models.temporal_cnn import conv_se_model

if __name__ == "__main__":
    model = conv_se_model(stack_length=2)
    model.summary()
