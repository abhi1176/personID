
import numpy as np
import os
import pandas as pd
import pickle

from utils import face_preprocess, face_model, palm_preprocess, palm_model

from collections import defaultdict
from glob import glob
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm


THRESHOLD = 0.5

def get_encodings(file):
    image = Image.open(file)
    image = image.convert('RGB')
    face = np.expand_dims(np.asarray(image, dtype=np.float64), axis=0)
    face = preprocess_input(face, version=2)
    yhat = model.predict(face)
    return yhat


if __name__ == "__main__":
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                    pooling='avg')
    train_df = pd.read_csv("datasets/train.csv", usecols=['face', 'label'])
    val_df = pd.read_csv("datasets/val.csv", usecols=['face', 'label'])
    df = pd.concat([train_df, val_df])
    df.drop_duplicates(inplace=True)
    if not os.path.exists("face_encodings.pkl"):
        print("[INFO] face_encodings.pkl not found!")
        encodings = defaultdict(list)
        print(df.head())
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i, :]
            try:
                encd = get_encodings(row.face)
            except Exception as e:
                print("[ERROR] Failed to get encodings for: {}".format(row.face))
                print(e)
            else:
                encodings[row.label].append(encd)
        encodings = [(label, encods) for label, encods in encodings.items()]
        with open("face_encodings.pkl", 'wb') as f:
            pickle.dump(encodings, f)
    else:
        with open("face_encodings.pkl", 'rb') as f:
            encodings = pickle.load(f)

    true_count = false_count = 0
    for i in tqdm(range(len(encodings))):
        for j in range(i, len(encodings)):
            for e1 in encodings[i][1]:
                for e2 in encodings[j][1]:
                    if i == j:
                        if cosine(e1, e2) < THRESHOLD:
                            true_count += 1
                        else:
                            false_count += 1
                    else:
                        if cosine(e1, e2) >= THRESHOLD:
                            true_count += 1
                        else:
                            false_count += 1

    print("[INFO] Accuracy (th:{}): {}"
          .format(THRESHOLD, true_count / (true_count + false_count)))
