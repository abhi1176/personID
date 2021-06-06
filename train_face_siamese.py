
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, Dropout,
                                     Flatten, Dense, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16


IMG_SHAPE = [224, 224, 3]
lr = 0.0001
epochs = 1
steps_per_epoch = 2000
batch_size = 32
embedding_dim = 64


def euclidean_distance(left_right_feats):
    # unpack the left_right_feats into separate lists
    (featsA, featsB) = left_right_feats
    # compute the sum of squared distances between the left_right_feats
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the left_right_feats
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def contrastive_loss(y_true, y_preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y_true = tf.cast(y_true, y_preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(y_preds)
    squaredMargin = K.square(K.maximum(margin - y_preds, 0))
    loss = K.mean(y_true * squaredPreds + (1 - y_true) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


def face_model():
    model = VGG16(include_top=False, input_shape=IMG_SHAPE, weights='imagenet')
    x = Flatten()(model.output)
    x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(embedding_dim, activation='relu')(x)
    return Model(model.inputs, x)


def face_siamese_model(learning_rate, loss):
    feature_extractor = face_model()
    left_input = Input(shape=IMG_SHAPE)
    right_input = Input(shape=IMG_SHAPE)
    left_feats = feature_extractor(left_input)
    right_feats = feature_extractor(right_input)
    output = Lambda(euclidean_distance)([left_feats, right_feats])
    if loss == "contrastive_loss":
        loss = contrastive_loss
    else:
        output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=[left_input, right_input], outputs=output)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss)
    return model


@tf.autograph.experimental.do_not_convert
def imgprcs(files, label):
    def read_img(image_file):
        img = tf.io.read_file(image_file)
        img = tf.image.decode_bmp(img, channels=3)
        img = tf.image.resize(img, IMG_SHAPE[:2])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        return img
    return (read_img(files[0]), read_img(files[1])), label


def get_train_dataset(train_csv, batch_size):
    df = pd.read_csv(train_csv, usecols=['face', 'label'])
    df.drop_duplicates(inplace=True)
    all_dataframe = df.join(df, lsuffix="_left", rsuffix="_right", how='cross')

    # Only keep pairs where left_img and right_img are different
    all_dataframe = all_dataframe[
            all_dataframe.face_left != all_dataframe.face_right]

    # Dataframe where left_img and right_img are of same person
    pos_dataframe = all_dataframe[
        all_dataframe.label_left == all_dataframe.label_right][
            ['face_left', 'face_right']]


    # Dataframe where left_img and right_img are of different person
    neg_dataframe = all_dataframe[
        all_dataframe.label_left != all_dataframe.label_right][
            ['face_left', 'face_right']]

    print("[INFO] Reading positive dataset_generator..")
    print("No. of positive samples:", len(pos_dataframe.values))
    pos_dataset = tf.data.Dataset.from_tensor_slices(pos_dataframe.values)
    pos_dataset = pos_dataset.map(lambda x: (x, 1)).shuffle(1000)

    print("[INFO] Reading negative dataset_generator..")
    print("No. of negative samples:", len(neg_dataframe.values))
    neg_dataset = tf.data.Dataset.from_tensor_slices(neg_dataframe.values)
    neg_dataset = neg_dataset.map(lambda x: (x, 0)).shuffle(10000)

    # This is to handle data imbalance between same-pairs and different-pairs
    sample_dataset = tf.data.experimental.sample_from_datasets(
        [pos_dataset, neg_dataset], weights=[0.5, 0.5]).repeat()
    sample_dataset = sample_dataset.map(imgprcs)
    sample_dataset = sample_dataset.batch(batch_size)
    sample_dataset = sample_dataset.prefetch(2)
    return sample_dataset


def get_val_dataset(train_csv, val_csv, batch_size, num_person=10):
    anchor_df = pd.read_csv(train_csv, usecols=['face', 'label'])

    # Select only a few persons for validation
    persons = np.unique(anchor_df.label)[:10]
    anchor_df = anchor_df[anchor_df.label.isin(persons)]
    anchor_df.drop_duplicates(inplace=True)

    val_df = pd.read_csv(val_csv, usecols=['face', 'label'])
    # Selection only a few persons for validation
    val_df = val_df[val_df.label.isin(persons)]
    val_df.drop_duplicates(inplace=True)

    all_df = anchor_df.join(val_df, lsuffix="_left", rsuffix="_right", how='cross')
    all_df = all_df[all_df.face_left != all_df.face_right]
    labels = np.where((all_df.label_left == all_df.label_right), 1, 0)
    all_df.drop(columns=["label_left", "label_right"], inplace=True)
    dataset = tf.data.Dataset.from_tensor_slices((all_df.values, labels))
    dataset = dataset.map(imgprcs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch-size", default=batch_size, type=int)
    parser.add_argument("-r", "--learning-rate", default=lr, type=float)
    parser.add_argument("-e", "--epochs", default=epochs, type=int)
    parser.add_argument("-s", "--steps-per-epoch", default=steps_per_epoch, type=int)
    parser.add_argument("-l", "--loss", default="contrastive_loss")
    args = parser.parse_args()

    train_dataset = get_train_dataset('datasets/train.csv', batch_size=args.batch_size)
    val_dataset = get_val_dataset("datasets/train.csv", "datasets/val.csv", batch_size=args.batch_size)

    print("[INFO] learning rate:", args.learning_rate)
    print("[INFO] epochs:", args.epochs)
    print("[INFO] batch size:", args.batch_size)
    print("[INFO] steps_per_epoch:", args.steps_per_epoch)

    save_model_as = "models/face_epochs{}_lr{}_batch{}_loss{}"
    save_model_as = save_model_as.format(args.epochs, args.learning_rate,
                                         args.batch_size, args.loss)

    model = face_siamese_model(args.learning_rate, args.loss)
    history = model.fit(train_dataset, epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        validation_data=val_dataset)
    print("[INFO] Saving the model to {}".format(save_model_as))
    os.makedirs(os.path.dirname(save_model_as), exist_ok=True)
    model.save(save_model_as)
