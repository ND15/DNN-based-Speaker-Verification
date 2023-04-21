import os

import librosa
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from audio_utils.preprocess import dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from models.baseline_model import model
from audio_utils.metrics import calculate_eer
from sklearn.model_selection import train_test_split

BATCH = 64
MODE = 'train'


def tf_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    print("Dataset Shapes\n")
    print(X_train.shape, X_test.shape, X_valid.shape)
    print(y_train.shape, y_test.shape, y_valid.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH, drop_remainder=True)
    train_dataset = train_dataset.prefetch(buffer_size=1000)

    print("Train Dataset")

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1000).batch(BATCH, drop_remainder=True)
    test_dataset = test_dataset.prefetch(buffer_size=1000)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    valid_dataset = valid_dataset.shuffle(buffer_size=1000).batch(BATCH, drop_remainder=True)
    valid_dataset = valid_dataset.prefetch(buffer_size=1000)

    return train_dataset, test_dataset, valid_dataset


filenames = glob.glob("/home/nikhil/Datasets/vox/vox1_indian/content/vox_indian/**/**/*.wav")
df = pd.read_csv("/home/nikhil/Datasets/vox/vox1_meta.csv", sep='\t')
X, y = dataset(filenames, df)
y = y.reshape((len(y), 1))

print(X.shape, y.shape)

print(np.unique(y))

oe = OneHotEncoder(sparse=True)
le = LabelEncoder()
y_le_encoded = le.fit_transform(y)
y_encoded = oe.fit_transform(y).toarray()
print(y, y_encoded.shape)

train_data, test_data, valid_data = tf_dataset(X, y_encoded)

keras.backend.clear_session()

for i in test_data.take(1).as_numpy_iterator():
    x_pred = i[0]

    y_pred = le.transform(oe.inverse_transform(i[1]).ravel())

    print(oe.inverse_transform(i[1]), "\t", y_pred)

print(x_pred.shape, y_pred.shape)

if MODE == 'train':
    m = model()

    checkpoints = keras.callbacks.ModelCheckpoint("./checkpoints/base_model_1",
                                                  monitor='val_loss',
                                                  verbose=0,
                                                  save_best_only=False,
                                                  save_weights_only=False,
                                                  mode='auto',
                                                  save_freq='epoch')

    tf_board = keras.callbacks.TensorBoard(log_dir="./logs/fit/base_model_1",
                                           histogram_freq=1,
                                           write_graph=True,
                                           write_images=True,
                                           update_freq='epoch',
                                           profile_batch=2,
                                           embeddings_freq=1)

    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    m.summary()

    for i in range(100):
        print(f'Epoch {i}/100')
        m.fit(train_data, batch_size=BATCH, verbose=1, validation_data=valid_data)
        # callbacks=[checkpoints, tf_board])

        m.save(f"ln_logs/epoch_{i}.h5", overwrite=True)

        print(m.evaluate(test_data))

elif MODE == 'test':
    m = tf.keras.models.load_model("logs/base_model_epoch_10.h5")
    print("\nPredicted values\n")
    pred = m.predict(x_pred)
    pred = le.transform(oe.inverse_transform(pred).ravel())

    confusion = confusion_matrix(y_pred, pred)

    eer = calculate_eer(confusion)

    print(confusion)

elif MODE == 'predict':
    m = tf.keras.models.load_model("logs/base_model_epoch_10.h5")
    print("\nPredicted Output")
    x_sample, sr = librosa.load("../testing/k_r.wav")
    x_sample = x_sample[np.newaxis, 32000:64000, np.newaxis]
    pred = m.predict(x_sample)
    pred = le.transform(oe.inverse_transform(pred))
    print(pred)
