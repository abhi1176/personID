
import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import soundfile as sf

from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Concatenate, BatchNormalization,
                                     Dense, Subtract, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from train_face_siamese import face_model
from train_palm_print_siamese import palm_print_model
from train_audio_siamese import audio_baseline_model


# Audio model params
filters = 128
audio_embedding_len = 64
audio_len = 48000
n_seconds = 3
SAMPLING_RATE = 16000
pad = True

face_img_shape = (224, 224, 3)
palm_img_shape = (224, 224, 3)

# Siamese model configurations
epochs = 1
batch_size = 32
learning_rate = 0.001
steps_per_epoch = 100
num_persons = 50


def cascade_siamese_model(learning_rate=learning_rate):
    f_model = face_model()
    for layer in f_model.layers:
        layer._name = layer.name + str("_face")
    # f_model.summary()

    p_model = palm_print_model()
    for layer in p_model.layers:
        layer._name = layer.name + str("_palm")
    # p_model.summary()

    a_model = audio_baseline_model(filters, audio_embedding_len, input_shape=(audio_len, 1))
    for layer in a_model.layers:
        layer._name = layer.name + str("_audio")
    # a_model.summary()

    merge_1 = Concatenate()([a_model.output, p_model.output])
    merge_1 = BatchNormalization()(merge_1)
    fc_1 = Dense(32, activation='relu')(merge_1)

    merge_2 = Concatenate()([f_model.output, fc_1])
    merge_2 = BatchNormalization()(merge_2)
    fc_2 = Dense(32, activation='relu')(merge_2)

    encoder = Model(inputs=[f_model.input, p_model.input, a_model.input],
                    outputs=fc_2, name='encoder')
    # encoder.summary()
    # plot_model(encoder, to_file='encoder.png')

    input_f1 = Input(f_model.input.shape[1:])
    input_f2 = Input(f_model.input.shape[1:])

    input_p1 = Input(p_model.input.shape[1:])
    input_p2 = Input(p_model.input.shape[1:])

    input_a1 = Input(a_model.input.shape[1:])
    input_a2 = Input(a_model.input.shape[1:])

    left_encoder = encoder([input_f1, input_p1, input_a1])
    right_encoder = encoder([input_f2, input_p2, input_a2])

    embedded_distance = Subtract(name='subtract_embeddings')([left_encoder, right_encoder])
    embedded_distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)),
                               name='euclidean_distance')(embedded_distance)
    output = Dense(1, activation='sigmoid', name='output')(embedded_distance)
    siamese = Model(inputs=[input_f1, input_p1, input_a1, input_f2, input_p2, input_a2],
                    outputs=output)
    siamese.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),
                    metrics=['accuracy'])
    # siamese.summary()
    return siamese


def read_face(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, face_img_shape[:2])
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, face_img_shape[:2])
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = img/255.
    return img


def read_palm_print(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, palm_img_shape[:2])
    img = img/255.
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, palm_img_shape[:2])
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img


def read_audio(wav_file, stochastic, seconds=n_seconds):
    sample, rate = sf.read(wav_file)
    assert (rate == SAMPLING_RATE)
    fragment_len = int(seconds * rate)
    # Choose a random sample of the file
    if stochastic:
        start_index = np.random.randint(0, max(len(sample)-fragment_len, 1))
    else:
        start_index = 0
    sample = sample[start_index:start_index+fragment_len]
    # Check for required length and pad if necessary
    if pad and len(sample) < fragment_len:
        less_timesteps = fragment_len - len(sample)
        if stochastic:
            # Stochastic padding, ensure sample length == fragment_len by appending a random number of 0s
            # before and the appropriate number of 0s after the sample
            less_timesteps = fragment_len - len(sample)
            before_len = np.random.randint(0, less_timesteps)
            after_len = less_timesteps - before_len
            sample = np.pad(sample, (before_len, after_len), 'constant')
        else:
            # Deterministic padding. Append 0s to reach fragment_len
            sample = np.pad(sample, (0, less_timesteps), 'constant')
    return np.expand_dims(sample, axis=-1)


def gen_fn(df, stochastic):
    def process():
        for idx, row in df.iterrows():
            face_left = read_face(row['face_left'])
            face_right = read_face(row['face_right'])
            palm_print_left = read_palm_print(row['palm_print_left'])
            palm_print_right = read_palm_print(row['palm_print_right'])
            audio_left = read_audio(row['audio_left'], stochastic)
            audio_right = read_audio(row['audio_right'], stochastic)
            yield ((face_left, palm_print_left, audio_left,
                    face_right, palm_print_right, audio_right),
                   row.label)
    return process


def get_train_dataset(train_csv, batch_size, num_persons):
    df = pd.read_csv(train_csv, usecols=['face', 'palm_print', 'audio', 'label'])
    persons = np.unique(df.label)[:num_persons]
    df = df[df.label.isin(persons)]
    all_df = df.join(df, lsuffix="_left", rsuffix="_right", how='cross')
    df.drop_duplicates(inplace=True)

    all_df['label'] = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df.drop(columns=['label_left', 'label_right'], inplace=True)

    # Dataframe where left_img and right_img are of same person
    pos_df = all_df[all_df.label == 1]
    print("No. of positive samples:", len(pos_df.values))

    # Dataframe where left_img and right_img are of different person
    neg_df = all_df[all_df.label == 0]
    num_neg_samples = pos_df.shape[0] # * 2
    neg_df = neg_df.sample(pos_df.shape[0])
    print("No. of negative samples:", len(neg_df.values))

    df = pd.concat([pos_df, neg_df]).sample(frac=1)  # Shuffle dataframe
    dataset = tf.data.Dataset.from_generator(gen_fn(df, True),
        output_shapes=(
            (face_img_shape, palm_img_shape, [n_seconds*SAMPLING_RATE, 1],
             face_img_shape, palm_img_shape, [n_seconds*SAMPLING_RATE, 1]),
            []),
        output_types=(
            (np.float32, np.float32, np.float32,
             np.float32, np.float32, np.float32),
            np.int32))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def get_val_dataset(val_csv, batch_size, num_persons):
    df = pd.read_csv(val_csv, usecols=['face', 'palm_print', 'audio', 'label'])
    persons = np.unique(df.label)[:num_persons]
    df = df[df.label.isin(persons)]
    all_df = df.join(df, lsuffix="_left", rsuffix="_right", how='cross')
    df.drop_duplicates(inplace=True)

    all_df['label'] = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df.drop(columns=['label_left', 'label_right'], inplace=True)

    # Dataframe where left_img and right_img are of same person
    pos_df = all_df[all_df.label == 1]
    print("No. of positive samples:", len(pos_df.values))

    # Dataframe where left_img and right_img are of different person
    neg_df = all_df[all_df.label == 0]
    num_neg_samples = pos_df.shape[0] # * 2
    neg_df = neg_df.sample(pos_df.shape[0])
    print("No. of negative samples:", len(neg_df.values))

    df = pd.concat([pos_df, neg_df]).sample(frac=1)  # Shuffle dataframe
    dataset = tf.data.Dataset.from_generator(gen_fn(df, True),
        output_shapes=(
            (face_img_shape, palm_img_shape, [n_seconds*SAMPLING_RATE, 1],
             face_img_shape, palm_img_shape, [n_seconds*SAMPLING_RATE, 1]),
            []),
        output_types=(
            (np.float32, np.float32, np.float32,
             np.float32, np.float32, np.float32),
            np.int32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", default=epochs, type=int)
    parser.add_argument("-s", "--steps-per-epoch", default=steps_per_epoch, type=int)
    parser.add_argument("-l", "--learning-rate", default=learning_rate, type=float)
    parser.add_argument("-b", "--batch-size", default=batch_size, type=int)
    parser.add_argument("-p", "--num-persons", default=num_persons, type=int)
    args = parser.parse_args()

    print("[INFO] Generating dataset..")
    train_dataset = get_train_dataset("datasets/train.csv", batch_size=32, num_persons=args.num_persons)
    val_dataset = get_val_dataset("datasets/val.csv", batch_size=32, num_persons=args.num_persons)

    print("[INFO] Generating model..")
    siamese_model = cascade_siamese_model(learning_rate=args.learning_rate)

    print("[INFO] Training model..")
    siamese_model.fit(train_dataset, epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch,
                      validation_data=val_dataset)
    save_as = "models/siamese_epoch{}_steps{}_lr{}_batch{}"
    save_as = save_as.format(args.epochs, args.steps_per_epoch,
                             args.learning_rate, args.batch_size)
    os.makedirs(os.path.dirname(save_as))

    print("[INFO] Saving the model to: {}".format(save_as))
    siamese_model.save(save_as)
