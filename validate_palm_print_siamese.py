
import tensorflow as tf
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from tensorflow.keras.models import load_model

IMG_SHAPE = [224, 224, 3]
threshold = 0.5

def contrastive_loss(y_true, y_preds):
    return True

@tf.autograph.experimental.do_not_convert
def imgprcs(files):
    def read_img(image_file):
        img = tf.io.read_file(image_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SHAPE[:2])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        return img
    return (read_img(files[0]), read_img(files[1])), -1


def get_val_dataset(train_csv, val_csv, batch_size, num_person=10):
    anchor_df = pd.read_csv(train_csv, usecols=['palm_print', 'label'])

    # Select only a few persons for validation
    persons = np.unique(anchor_df.label)[:10]
    anchor_df = anchor_df[anchor_df.label.isin(persons)]
    anchor_df.drop_duplicates(inplace=True)

    val_df = pd.read_csv(val_csv, usecols=['palm_print', 'label'])
    # Selection only a few persons for validation
    val_df = val_df[val_df.label.isin(persons)]
    val_df.drop_duplicates(inplace=True)

    all_df = anchor_df.join(val_df, lsuffix="_left", rsuffix="_right", how='cross')
    all_df = all_df[all_df.palm_print_left != all_df.palm_print_right]
    labels = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df.drop(columns=["label_left", "label_right"], inplace=True)
    dataset = tf.data.Dataset.from_tensor_slices(all_df.values)
    dataset = dataset.map(imgprcs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset, labels


def calc_accuracy(y_true, y_pred, threshold):
    return np.mean(np.equal((y_pred < threshold), y_true))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-t", "--threshold", default=threshold, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    args = parser.parse_args()
    model = load_model(args.model, custom_objects={
        'loss': contrastive_loss, 'contrastive_loss': contrastive_loss})
    val_dataset, y_true = get_val_dataset("datasets/train.csv", "datasets/val.csv",
                                          batch_size=args.batch_size)
    y_pred = model.predict(val_dataset).flatten()
    print("Accuracy: {}".format(calc_accuracy(y_true, y_pred, args.threshold)))
