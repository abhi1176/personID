import cv2
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import os

from datetime import datetime
from random import shuffle

from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

INPUT_SHAPE = (224, 224, 3)
IMAGES_DIR = os.path.join("datasets", "CASIA-PalmprintV1")
num_classes = 300

def train_test_split():
    labels = sorted(os.listdir(IMAGES_DIR))[:num_classes]
    train_files = []
    eval_files = []
    for label in labels:
        folder = os.path.join(IMAGES_DIR, label)
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".jpg")]
        shuffle(files)
        split_idx = int(len(files) * 0.7)
        train_files.extend(files[:split_idx])
        eval_files.extend(files[split_idx:])
    return train_files, eval_files, labels

train_files, eval_files, labels = train_test_split()
print("Train files: {}".format(len(train_files)))
print("Eval files: {}".format(len(eval_files)))
print("Labels: {}".format(len(labels)))


def palm_model():
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    for layer in vgg16.layers:
        layer.trainable = False
    x = vgg16.output
    x = layers.Flatten()(x) # Flatten dimensions to for use in FC layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(len(labels), activation='softmax')(x) # Softmax for multiclass
    return Model(inputs=vgg16.input, outputs=x)


class PersonIDSequence(Sequence):

    def __init__(self, files, labels, batch_size, extract_palm=False):
        self.files = files
        self.labels = labels
        shuffle(self.files)
        self.num_labels = len(self.labels)
        self.batch_size = batch_size
        self.extract_palm = extract_palm

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def load_palm_print(self, image_file):
        if INPUT_SHAPE[2] == 1:
            image = cv2.imread(image_file, 0)
        else:
            image = cv2.imread(image_file)
        if self.extract_palm:
            image = extract_palm_from_img(image)
        try:
            image = cv2.resize(image, INPUT_SHAPE[:2])
        except:
            print("image_file:", image_file)
        image = image * 1./255
        return image

    def __getitem__(self, idx):
        X = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = [os.path.basename(os.path.dirname(file)) for file in X ]
        palm_prints = np.array(list(map(self.load_palm_print, X)))
        y_indices = [to_categorical(self.labels.index(i), num_classes=self.num_labels)
                     for i in y]
        return palm_prints, np.array(y_indices)

    def on_epoch_end(self):
        shuffle(self.files)


model = palm_model()
model.summary()

train_ds = PersonIDSequence(train_files, labels, batch_size=64)
eval_ds = PersonIDSequence(eval_files, labels, batch_size=64)

epochs = 300
lr = 0.001

model_dir = os.path.join("models", datetime.now().strftime('%Y%m%d%H%M'))
os.makedirs(model_dir, exist_ok=True)

# Each epoch has 56 batches. Save the model for every 50th batch
mc = ModelCheckpoint(os.path.join(model_dir, "palmnet_e{epoch:03d}"), save_freq=56*50)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr))
history = model.fit(train_ds, epochs=epochs, validation_data=eval_ds, callbacks=[mc])

model.predict(eval_ds)
models.save(os.path.join(models_dir, "final_model"))
