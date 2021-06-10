
import numpy as np

from keras_vggface.vggface import VGGFace
from PIL import Image
from skimage.transform import resize
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


def face_model():
    return VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                   pooling='avg')

def palm_print_model():
    model_file = "models/palm_features_model.h5"
    return load_model(model_file)

def palm_print_diff_model():
    model_file = "models/palm_diff_model.h5"
    return load_model(model_file)