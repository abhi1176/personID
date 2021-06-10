
import os
import pandas as pd
import pickle

from collections import defaultdict
from scipy.spatial.distance import cosine
from tqdm import tqdm

from models_utils import face_model
from preproc_utils import face_preprocess


THRESHOLD = 0.5

def get_encodings(file):
    face = face_preprocess(file)
    yhat = model.predict(face)
    return yhat


if __name__ == "__main__":
    model = face_model()
    train_df = pd.read_csv("datasets/train.csv", usecols=['face', 'label'])
    val_df = pd.read_csv("datasets/val.csv", usecols=['face', 'label'])
    df = pd.concat([train_df, val_df])
    df.drop_duplicates(inplace=True)
    if not os.path.exists("encodings/face_encodings.pkl"):
        os.makedirs("encodings", exist_ok=True)
        print("[INFO] encodings/face_encodings.pkl not found!")
        encodings = defaultdict(list)
        print("[INFO] Extracting face features")
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
        with open("encodings/face_encodings.pkl", 'wb') as f:
            pickle.dump(encodings, f)
    else:
        with open("encodings/face_encodings.pkl", 'rb') as f:
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
