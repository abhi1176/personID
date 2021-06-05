
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import cvlib as cv

from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
from tensorflow.keras.utils import to_categorical


face_img_size = (224, 224)
palm_img_size = (224, 224)
maxlen = 200  # Maximum number of strokes
numcep = 13
nfilt = 26
nfft = 512
threshold = 200


def features_from_txt(text_file):
    data = np.loadtxt(text_file, skiprows=1, dtype=np.float32)
    # Column-wise min-max scaling
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    if len(data) < maxlen:
        pad = maxlen - len(data)
        data = np.pad(data, ((0, pad), (0, 0)))
    data = data[:maxlen, :]
    return data


def features_from_audio(audio_file, secs=0.1):
    def envelop(signal, rate, threshold):
        mask = []
        y = pd.Series(signal).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    rate, signal = wavfile.read(audio_file)
    mask = envelop(signal, rate, threshold)
    signal = signal[mask]
    step = int(rate*secs)
    rand_idx = np.random.randint(0, signal.shape[0]-step)
    sample = signal[rand_idx:rand_idx+step]
    sample = mfcc(sample, rate, numcep=numcep, nfilt=nfilt, nfft=nfft)
    sample = np.expand_dims(sample, axis=-1)
    return sample.astype(np.float32)


index = 0
def read_face_img(image_file):
    # global index
    image = cv2.imread(image_file)
    # faces, confidences = cv.detect_face(image)
    # confidences = np.array(confidences)
    # idx = np.argmax(confidences)
    # x1, y1, x2, y2 = faces[idx]
    # face = image[x1:x2, y1:y2]
    # cv2.imwrite("face_{}.jpg".format(index), face)
    # index += 1
    face = image
    face = cv2.resize(face, face_img_size)
    face = face * 1./255
    return face.astype(np.float32)


def read_palm_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, palm_img_size)
    image = image * 1./255
    return image.astype(np.float32)


def generate_data(csv_file):
    def process():
        df = pd.read_csv(csv_file)
        unique_labels = list(np.unique(df.label))
        num_labels = len(unique_labels)
        total_rows = df.shape[0]
        row_idx = 0
        while True:
            if row_idx % total_rows == 0:
                row_idx = 0
                df.sample(frac=1)  # Shuffle dataframe
            row = df.iloc[row_idx, :]
            row_idx += 1
            face = read_face_img(row['face'])
            palm_print = read_palm_img(row['palm_print'])
            signature = features_from_txt(row['signature'])
            audio = features_from_audio(row['audio'], 1)
            label = to_categorical(unique_labels.index(row['label']), num_classes=num_labels)
            yield (face, palm_print, signature, audio), label
    return process


def data_generator(csv_file, batch_size):
    gen = generate_data(csv_file)
    dataset = tf.data.Dataset.from_generator(gen,
        output_types=(
            (tf.float32, tf.float32, tf.float32, tf.float32),
            tf.float32
        ),
        output_shapes=(
            ((224, 224, 3), (224, 224, 3), (200, 7), (99, 13, 1)),
             20
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
