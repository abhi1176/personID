
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import soundfile as sf

from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, SpatialDropout1D, MaxPool1D,
    GlobalMaxPool1D, Dense, Subtract, Lambda)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16


IMG_SHAPE = [224, 224, 3]
lr = 0.0001
epochs = 1
steps_per_epoch = 2000
batch_size = 32
SAMPLING_RATE = 16000

##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
filters = 128
embedding_dimension = 64
dropout = 0.0
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
pad = True
num_epochs = 50
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5


# stochastic is True for training, else False
def get_fragment(wav_file, stochastic, seconds=n_seconds):
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


def audio_baseline_model(filters, embedding_dimension, input_shape=None, dropout=0.05):
    encoder = Sequential()
    encoder.add(Input(shape=input_shape))
    encoder.add(Conv1D(filters, 32, padding='same', activation='relu'))
    encoder.add(BatchNormalization())
    encoder.add(SpatialDropout1D(dropout))
    encoder.add(MaxPool1D(4, 4))

    # Further convs
    encoder.add(Conv1D(2*filters, 3, padding='same', activation='relu'))
    encoder.add(BatchNormalization())
    encoder.add(SpatialDropout1D(dropout))
    encoder.add(MaxPool1D())

    encoder.add(Conv1D(3 * filters, 3, padding='same', activation='relu'))
    encoder.add(BatchNormalization())
    encoder.add(SpatialDropout1D(dropout))
    encoder.add(MaxPool1D())

    encoder.add(Conv1D(4 * filters, 3, padding='same', activation='relu'))
    encoder.add(BatchNormalization())
    encoder.add(SpatialDropout1D(dropout))
    encoder.add(MaxPool1D())

    encoder.add(GlobalMaxPool1D())

    encoder.add(Dense(embedding_dimension))
    return encoder


def build_siamese_net(encoder, input_shape, distance_metric='uniform_euclidean'):
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    if distance_metric == 'weighted_l1':
        # This is the distance metric used in the original one-shot paper
        # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
        embedded_distance = Subtract()([encoded_1, encoded_2])
        embedded_distance = Lambda(lambda x: K.abs(x))(embedded_distance)
        output = Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'uniform_euclidean':
        # Simpler, no bells-and-whistles euclidean distance
        # Still apply a sigmoid activation on the euclidean distance however
        embedded_distance = Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
        # Sqrt of sum of squares
        embedded_distance = Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)), name='euclidean_distance'
        )(embedded_distance)
        output = Dense(1, activation='sigmoid')(embedded_distance)

    return Model(inputs=[input_1, input_2], outputs=output)


def get_model():
    ################
    # Define model #
    ################
    # input_length = int(SAMPLING_RATE * n_seconds / downsampling)
    input_length = int(SAMPLING_RATE * n_seconds)

    encoder = audio_baseline_model(filters, embedding_dimension,
                                   input_shape=(input_length, 1),
                                   dropout=dropout)
    siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
    opt = Adam(clipnorm=1.)
    siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # plot_model(siamese, show_shapes=True, to_file='audio_siamese.png')
    siamese.summary()
    return siamese


def generator(df, labels, stochastic):
    def process():
        for idx, row in df.iterrows():
            try:
                label = labels[idx]
            except:
                label = labels
            yield (get_fragment(row.audio_left, stochastic),
                   get_fragment(row.audio_right, stochastic)), label
    return process


def get_train_dataset(train_csv, batch_size):
    df = pd.read_csv(train_csv, usecols=['audio', 'label'])
    df.drop_duplicates(inplace=True)
    all_dataframe = df.join(df, lsuffix="_left", rsuffix="_right", how='cross')

    # Only keep pairs where left_img and right_img are different
    all_dataframe = all_dataframe[
            all_dataframe.audio_left != all_dataframe.audio_right]

    # Dataframe where left_img and right_img are of same person
    pos_dataframe = all_dataframe[
        all_dataframe.label_left == all_dataframe.label_right][
            ['audio_left', 'audio_right']]


    # Dataframe where left_img and right_img are of different person
    neg_dataframe = all_dataframe[
        all_dataframe.label_left != all_dataframe.label_right][
            ['audio_left', 'audio_right']]

    neg_dataframe = neg_dataframe.sample(pos_dataframe.shape[0])

    print("[INFO] Reading positive dataset_generator..")
    print("No. of positive samples:", len(pos_dataframe.values))
    pos_dataset = tf.data.Dataset.from_generator(generator(pos_dataframe, 1, True),
        output_shapes=(([n_seconds*SAMPLING_RATE, 1], [n_seconds*SAMPLING_RATE, 1]),
                       []),
        output_types=((np.float32, np.float32), np.int32))
    # pos_dataset = tf.data.Dataset.from_tensor_slices(pos_dataframe.values)
    # pos_dataset = pos_dataset.map(lambda x: (x, 1)).shuffle(1000)

    print("[INFO] Reading negative dataset_generator..")
    print("No. of negative samples:", len(neg_dataframe.values))
    neg_dataset = tf.data.Dataset.from_generator(generator(neg_dataframe, 0, True),
        output_shapes=(([n_seconds*SAMPLING_RATE, 1], [n_seconds*SAMPLING_RATE, 1]),
                       []),
        output_types=((np.float32, np.float32), np.int32))
    # neg_dataset = tf.data.Dataset.from_tensor_slices(neg_dataframe.values)
    # neg_dataset = neg_dataset.map(lambda x: (x, 0)).shuffle(10000)

    # This is to handle data imbalance between same-pairs and different-pairs
    sample_dataset = tf.data.experimental.sample_from_datasets(
        [pos_dataset, neg_dataset], weights=[0.5, 0.5]).repeat()
    # sample_dataset = sample_dataset.map(lambda x, y: audio_process(x, y, True))
    sample_dataset = sample_dataset.batch(batch_size)
    sample_dataset = sample_dataset.prefetch(2)
    return sample_dataset


def get_val_dataset(train_csv, val_csv, batch_size, num_person=10):
    anchor_df = pd.read_csv(train_csv, usecols=['audio', 'label'])

    # Select only a few persons for validation
    persons = np.unique(anchor_df.label)[:10]
    anchor_df = anchor_df[anchor_df.label.isin(persons)]
    anchor_df.drop_duplicates(inplace=True)

    val_df = pd.read_csv(val_csv, usecols=['audio', 'label'])
    # Selection only a few persons for validation
    val_df = val_df[val_df.label.isin(persons)]
    val_df.drop_duplicates(inplace=True)

    all_df = anchor_df.join(val_df, lsuffix="_left", rsuffix="_right", how='cross')
    all_df = all_df[all_df.audio_left != all_df.audio_right]
    labels = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df.drop(columns=["label_left", "label_right"], inplace=True)
    # dataset = tf.data.Dataset.from_tensor_slices((all_df.values, labels))
    dataset = tf.data.Dataset.from_generator(generator(all_df, labels, False),
        output_shapes=(([n_seconds*SAMPLING_RATE, 1], [n_seconds*SAMPLING_RATE, 1]),
                       []),
        output_types=((np.float32, np.float32), np.int32))
    # dataset = dataset.map(lambda x, y: audio_process(x, y, False))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch-size", default=batch_size, type=int)
    parser.add_argument("-l", "--learning-rate", default=lr, type=float)
    parser.add_argument("-e", "--epochs", default=epochs, type=int)
    parser.add_argument("-s", "--steps-per-epoch", default=steps_per_epoch, type=int)
    args = parser.parse_args()

    train_dataset = get_train_dataset('datasets/train.csv', batch_size=args.batch_size)
    val_dataset = get_val_dataset("datasets/train.csv", "datasets/val.csv", batch_size=args.batch_size)

    # df = pd.read_csv('train.csv', usecols=['audio'])
    # num_audios_in_train = df.drop_duplicates().shape[0]
    # steps_per_epoch = (num_audios_in_train**2)//args.batch_size

    # df = pd.read_csv('val.csv', usecols=['audio'])
    # num_audios_in_val = df.drop_duplicates().shape[0]
    # validation_steps = (num_audios_in_val**2)//args.batch_size
    print("[INFO] learning rate:", args.learning_rate)
    print("[INFO] epochs:", args.epochs)
    print("[INFO] batch size:", args.batch_size)
    print("[INFO] steps_per_epoch:", args.steps_per_epoch)

    save_model_as = "models/audio_epochs{}_lr{}_batch{}"
    model_output = save_model_as.format(args.epochs, lr, args.batch_size)

    model = get_model()

    history = model.fit(train_dataset, epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        validation_data=val_dataset)
    print("[INFO] Saving the model to {}".format(model_output))
    model.save(model_output)
