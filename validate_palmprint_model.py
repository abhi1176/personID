
import cv2
import os
import numpy as np
import pandas as pd
import pickle
import random

from collections import defaultdict
from scipy.spatial.distance import cosine
from tqdm import tqdm

from models_utils import palm_print_model, palm_print_diff_model
from preproc_utils import palm_print_preprocess


THRESHOLD = 0.5


def get_encodings(model, file):
    palm_print = palm_print_preprocess(file)
    yhat = model.predict(palm_print)
    return yhat


if __name__ == "__main__":
    model = palm_print_model()
    train_df = pd.read_csv("datasets/train.csv", usecols=['palm_print', 'label'])
    val_df = pd.read_csv("datasets/val.csv", usecols=['palm_print', 'label'])
    df = pd.concat([train_df, val_df])
    df.drop_duplicates(inplace=True)
    if not os.path.exists("encodings/palm_print_encodings.pkl"):
    # if 1:
        os.makedirs("encodings", exist_ok=True)
        print("[INFO] encodings/palm_print_encodings.pkl not found!")
        encodings = defaultdict(list)
        print("[INFO] Extracting palm_print features")
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i, :]
            try:
                encd = get_encodings(model, row.palm_print)
            except Exception as e:
                print("[ERROR] Failed to get encodings for: {}".format(row.palm_print))
                print(e)
            else:
                encodings[row.label].append(encd)
        encodings = [(label, encods) for label, encods in encodings.items()]
        with open("encodings/palm_print_encodings.pkl", 'wb') as f:
            pickle.dump(encodings, f)
    else:
        with open("encodings/palm_print_encodings.pkl", 'rb') as f:
            encodings = pickle.load(f)

    true_count = false_count = 0
    print("[INFO] Started comparing...")
    model = palm_print_diff_model()
    model.summary()
    random_encodings = random.sample(encodings, 20)
    for i, j in random_encodings:
        print(i)
    exit()
    for i in tqdm(range(len(random_encodings))):
        for j in tqdm(range(i+1, i+2)):
            for e1 in random_encodings[i][1]:
                for e2 in random_encodings[j][1]:
                    e_diff = e1 - e2
                    distance = model.predict(e_diff)
                    # print(distance)
                    distance = distance.flatten()[0]
                    if i == j:
                        if distance < THRESHOLD:
                            true_count += 1
                        else:
                            false_count += 1
                    else:
                        if distance >= THRESHOLD:
                            true_count += 1
                        else:
                            false_count += 1
        print("[INFO] Accuracy (th:{}): {}"
              .format(THRESHOLD, true_count / (true_count + false_count)))
