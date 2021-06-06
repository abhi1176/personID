
import tensorflow as tf
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from tensorflow.keras.models import load_model


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


def generator(df, stochastic):
    def process():
        for idx, row in df.iterrows():
            yield (get_fragment(row.audio_left, stochastic),
                   get_fragment(row.audio_right, stochastic)), row.label
    return process


def get_val_dataset(train_csv, val_csv, batch_size, num_persons=10):
    anchor_df = pd.read_csv(train_csv, usecols=['audio', 'label'])

    # Select only a few persons for validation
    persons = np.unique(anchor_df.label)[:num_persons]
    anchor_df = anchor_df[anchor_df.label.isin(persons)]
    anchor_df.drop_duplicates(inplace=True)

    val_df = pd.read_csv(val_csv, usecols=['audio', 'label'])
    # Selection only a few persons for validation
    val_df = val_df[val_df.label.isin(persons)]
    val_df.drop_duplicates(inplace=True)

    all_df = anchor_df.join(val_df, lsuffix="_left", rsuffix="_right", how='cross')
    all_df = all_df[all_df.audio_left != all_df.audio_right]
    labels = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df['label'] = -1
    all_df.drop(columns=["label_left", "label_right"], inplace=True)

    dataset = tf.data.Dataset.from_generator(generator(all_df, False),
        output_shapes=(([n_seconds*SAMPLING_RATE, 1], [n_seconds*SAMPLING_RATE, 1]),
                       []),
        output_types=((np.float32, np.float32), np.int32))
    # dataset = dataset.map(lambda x, y: audio_process(x, y, False))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def calc_accuracy(y_true, y_pred, threshold):
    return np.mean(np.equal((y_pred < threshold), y_true))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-t", "--threshold", default=threshold, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-p", "--num-persons", default=10, type=int)
    args = parser.parse_args()
    model = load_model(args.model)
    val_dataset, y_true = get_val_dataset("datasets/train.csv", "datasets/val.csv",
                                          batch_size=args.batch_size,
                                          num_persons=args.num_persons)
    y_pred = model.predict(val_dataset).flatten()
    print("Accuracy: {}".format(calc_accuracy(y_true, y_pred, args.threshold)))
