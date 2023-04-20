import glob
import librosa
import pandas as pd
import soundfile as sf
import tensorflow as tf
import numpy as np
from audio_utils.utils import MelSpec
from hparams import hparams

SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_LENGTH = 256
PATCH_SIZE = 32000
EPOCH = 20
BATCH = 8
SAMPLE_STRIDE = 1000


def prepare_audio_and_labels(files):
    audio_files = []
    labels = []

    for i in files:
        audio_files.append(i)
        labels.append(i.split('\\')[-3])

    return audio_files, labels


def map_labels(labels, dataframe):
    names = []
    df_labels = dataframe.copy()
    df_labels['VoxCeleb1 ID'] = df_labels['VoxCeleb1 ID'].astype('str')
    df_labels['VGGFace1 ID'] = df_labels['VGGFace1 ID'].astype('str')
    df_labels = df_labels[df_labels['VoxCeleb1 ID'].isin(set(labels))]
    df_labels = df_labels[['VoxCeleb1 ID', 'VGGFace1 ID']]

    for i in labels:
        values = df_labels[df_labels['VoxCeleb1 ID'] == i].values
        names.append(values[0][-1])

    return np.asarray(names)


def get_length(audio_files):
    audio_lens = []
    for i in audio_files:
        audio, sr = sf.read(i)
        audio_lens.append(len(audio))
        print(i)
    return np.asarray(audio_lens)


def create_dataset(files, labels, dataset_type='signal'):
    mel = MelSpec(hparams)
    total_len = 32000

    mels = []
    ids = []

    for file_name, label in zip(files, labels):
        data, samplerate = librosa.load(file_name)
        if dataset_type == "signal":
            mels.append(data)
            ids.append(label)
        else:
            mel_spectrogram = mel.spectrogram(data.astype('float32'))
            mels.append(mel_spectrogram.numpy().T)
            ids.append(label)
        print(f"Processed {file_name}")

    return np.asarray(mels), np.asarray(ids)


def dataset(filenames, df):
    audio_files, labels = prepare_audio_and_labels(filenames)

    names = map_labels(labels, df)

    print(len(audio_files))

    random_indices = np.random.choice(len(audio_files), 275, replace=False)

    mels, ids = create_dataset(np.asarray(audio_files)[random_indices],
                               np.asarray(names)[random_indices])
    X, y = [], []
    for mel, l_id in zip(mels, ids):
        starts = np.random.randint(0, mel.shape[0] - PATCH_SIZE, (mel.shape[0] - PATCH_SIZE) // SAMPLE_STRIDE)
        for start in starts:
            end = start + PATCH_SIZE
            X.append(mel[start:end])
            y.append(l_id)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


if __name__ == "__main__":
    f = glob.glob("D:/Downloads/Vox/vox1_indian/content/vox_indian/**/**/*.wav")
    d = pd.read_csv("D:/Downloads/Vox/vox1_meta.csv", sep='\t')
    X, y = dataset(f, d)
    print(X.shape)
    print(X[0])
    print(np.unique(y), np.unique(y).shape)
