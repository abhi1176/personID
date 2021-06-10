
import numpy as np
import cv2

from keras_vggface.utils import preprocess_input
from PIL import Image

# Face
def face_preprocess(file, size=(224, 224)):
    image = cv2.imread(file)[::-1]
    image = cv2.resize(image, size)
    face = np.expand_dims(image, axis=0)
    face = preprocess_input(face, version=2)
    return face

# Palm
def palm_print_preprocess(file, size=(90, 90)):
    palm_print = cv2.imread(file, 0)
    palm_print = cv2.resize(palm_print, size)
    palm_print = palm_print/255.
    palm_print = np.expand_dims(np.asarray(palm_print, dtype=np.float64), axis=(0, -1))
    return palm_print
