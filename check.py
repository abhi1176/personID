
import os
import tensorflow as tf
from random import shuffle
from tensorflow.keras.utils import Sequence, to_categorical


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


ds = tf.data.Dataset.from_tensor_slices(train_files)
def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    label = to_categorical(labels.index(label), num_classes=len(labels))
    return image, label
ds = ds.map(parse_image)
ds = ds.batch(32)
