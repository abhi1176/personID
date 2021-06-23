
import cv2
import json
import math
import numpy as np
import os
import pandas as pd

from keras_vggface.utils import preprocess_input
from mtcnn.mtcnn import MTCNN
from python_speech_features import mfcc
from scipy.io import wavfile
from tensorflow.keras.utils import Sequence, to_categorical

face_detector = MTCNN()

def extract_face_from_img(image):
    faces = face_detector.detect_faces(image)
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face


def extract_palm_from_img(image):
    image = np.rot90(image, 3)  # Rotate 90 degree clockwise
    h, w = image.shape
    img = np.zeros((h + 160, w + 160), np.uint8)  # Pad the image by 80 pixes on 4 sides
    img[80:-80, 80:-80] = image
    # Apply GaussionBlur to remove noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Binary + OTSU thresholding to generate Black-White image
    # White pixels denote the palm and back pixels denote the background
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Section 2
    M = cv2.moments(th)
    h, w = img.shape
    # Get centroid of the white pixels
    x_c = M['m10'] // M['m00']
    y_c = M['m01'] // M['m00']

    # Apply Erosion to remove noise
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]).astype(np.uint8)
    erosion = cv2.erode(th, kernel, iterations=1)
    boundary = th - erosion

    cnt, _ = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    areas = [cv2.contourArea(c) for c in cnt]
    max_index = np.argmax(areas)
    cnt = cnt[max_index]

    img_cnt = cv2.drawContours(img_c, [cnt], 0, (255, 0, 0), 2)

    cnt = cnt.reshape(-1, 2)
    left_id = np.argmin(cnt.sum(-1))
    cnt = np.concatenate([cnt[left_id:, :], cnt[:left_id, :]])

    # Section 3
    dist_c = np.sqrt(np.square(cnt-[x_c, y_c]).sum(-1))
    f = np.fft.rfft(dist_c)
    cutoff = 15
    f_new = np.concatenate([f[:cutoff], 0*f[cutoff:]])
    dist_c_1 = np.fft.irfft(f_new)

    # Section 4
    eta = np.square(np.abs(f_new)).sum()/np.square(np.abs(f)).sum()
    # print('Power Retained: {:.4f}{}'.format(eta*100, '%'))

    # Section 5
    derivative = np.diff(dist_c_1)
    sign_change = np.diff(np.sign(derivative))/2

    # Section 6
    minimas = cnt[np.where(sign_change>0)[0]]
    v1, v2 = minimas[-1], minimas[-3]

    theta = np.arctan2((v2-v1)[1], (v2-v1)[0])*180/np.pi
    R = cv2.getRotationMatrix2D((int(v2[0]), int(v2[1])), theta, 1)

    img_r = cv2.warpAffine(img, R, (w, h))
    v1 = (R[:, :2] @ v1 + R[:, -1]).astype(np.int)
    v2 = (R[:, :2] @ v2 + R[:, -1]).astype(np.int)

    ux = v1[0]
    uy = v1[1] + (v2-v1)[0]//3
    lx = v2[0]
    ly = v2[1] + 4*(v2-v1)[0]//3

    palm = img_r[uy:ly, ux:lx]
    return palm



def PersonIDSequence(csv_file, batch_size, config_file="config.json",
                     extract_face=False, extract_palm=False):

    def load_face(img_file):
        image = cv2.imread(os.path.join(relative_path, img_file))[::-1]  # Converting BGR to RGB
        if extract_face:
            image = extract_face_from_img(image)
        image = cv2.resize(image, config['face_shape'])
        image = image.astype(np.float32)
        image = preprocess_input(image, version=2)
        return image

    def load_palm_print(image_file):
        image = cv2.imread(os.path.join(relative_path, image_file))  # Read as Gray
        if extract_palm:
            image = extract_palm_from_img(image)
        image = cv2.resize(image, config['palm_shape'])
        # image = np.expand_dims(image, axis=-1)
        image = image * 1./255
        return image

    def load_signature(text_file):
        data = np.loadtxt(os.path.join(relative_path, text_file), skiprows=1, dtype=np.float32)
        # Column-wise min-max scaling
        diff = data.max(axis=0) - data.min(axis=0)
        diff = np.where(diff == 0, 1, diff)  # To handle division-by-zero error
        data = (data - data.min(axis=0)) / diff
        # Smoothing by rolling-window-mean-subtraction
        for i in range(data.shape[1]):
            data[:, i] -= pd.Series(data[:, i]).rolling(
                window=config['rolling_window'], center=True).mean()
        if len(data) < config['max_strokes']:
            pad = config['max_strokes'] - len(data)
            data = np.pad(data, ((0, pad), (0, 0)))  # Pad at the bottom
        else:
            n = np.linspace(0, len(data)-1, config['max_strokes'],
                            dtype=np.int32)
            data = data[n]
        # data = np.expand_dims(data, axis=-1)
        return data

    def load_audio(audio_file):
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
        rate, signal = wavfile.read(os.path.join(relative_path, audio_file))
        # mask = envelop(signal, rate, config['audio_clean_threshold'])
        # signal = signal[mask]
        step = int(rate*config['audio_seconds'])
        rand_idx = np.random.randint(0, signal.shape[0]-step)
        sample = signal[rand_idx:rand_idx+step]
        sample = mfcc(sample, rate,
                      numcep=config['audio_numcep'],
                      nfilt=config['audio_nfilt'],
                      nfft=config['audio_nfft'])
        sample = np.expand_dims(sample, axis=-1)
        return sample.astype(np.float32)

    relative_path = ".."
    df = pd.read_csv(csv_file, index_col=0).sample(frac=1)  # Shuffle
    labels = list(np.unique(df.label))
    num_labels = len(labels)
    batch_size = batch_size
    extract_face = extract_face
    extract_palm = extract_palm
    with open(config_file) as file:
        config = json.load(file)

    print("Loading all faces...")
    face_by_file = {file: load_face(file) for file in np.unique(df.face)}

    print("Loading all palm_prints...")
    palm_print_by_file = {file: load_palm_print(file)
                          for file in np.unique(df.palm_print)}
    print("Loading all audios...")
    audio_by_file = {file: load_audio(file) for file in np.unique(df.audio)}
    # signature_by_file = {file: load_signature(file) for file in np.unique(df.signature)}

    idx = 0
    n_rows = df.shape[0]

    def process():
        batch = df.iloc[idx * batch_size:(idx + 1) * batch_size]
        y = batch.pop('label')
        X = batch
        faces = np.array([face_by_file[file] for file in X.face])
        palm_prints = np.array([palm_print_by_file[file] for file in X.palm_print])
        audios = np.array([audio_by_file[file] for file in X.audio])
        y_indices = [to_categorical(labels.index(i), num_classes=num_labels)
                     for i in y.values]
        # yield (faces, palm_prints, audios, signatures), np.array(y_indices)
        yield (faces, palm_prints, audios), np.array(y_indices).astype(np.float32)
        idx = (idx + 1) % n_rows

