
import itertools
import numpy as np
import os
import pandas as pd

from random import shuffle
from glob import glob
from pprint import pprint
from sklearn.model_selection import train_test_split


# train_percent = 0.66
# val_percent = 0.5

train_percent = 0.7
val_percent = 1

num_persons = 300

data_dir = "datasets"

face_dir = os.path.join(data_dir, "FaceCropped")
# palm_dir = os.path.join(data_dir, "PalmCropped")
palm_dir = os.path.join(data_dir, "CASIA-PalmprintV1")
sign_dir = os.path.join(data_dir, "signature")
audio_dir = os.path.join(data_dir, "voxceleb1_wavfile")  # 193 <-- 192 149 <-- 148

face_persons = sorted(os.listdir(face_dir))[:num_persons]
palm_persons = sorted(os.listdir(palm_dir))[:num_persons]
sign_persons = sorted(os.listdir(sign_dir))[:num_persons]
audio_persons = sorted(os.listdir(audio_dir))[:num_persons]

print(palm_persons)

assert face_persons == palm_persons, "People are not same in face and plam directories"
assert sign_persons == audio_persons, "People are not same in sign and audio directories"
assert sign_persons == face_persons, "People are not same in sign and face directories"

shuffle(face_persons)

train_rows = []
val_rows = []
test_rows = []

for person in face_persons:
	faces = list(glob(os.path.join(face_dir, person, "*").replace("\\", "/")))
	faces = list(map(lambda x: x.replace("\\", "/"), faces))
	shuffle(faces)
	split1 = int(len(faces)*train_percent)
	train_faces, val_faces = faces[:split1], faces[split1:]
	split2 = int(len(val_faces)*val_percent)
	val_faces, test_faces = val_faces[:split2], val_faces[split2:]

	palmprints = list(glob(os.path.join(palm_dir, person, "*").replace("\\", "/")))
	palmprints = list(map(lambda x: x.replace("\\", "/"), palmprints))
	shuffle(palmprints)
	split1 = int(len(palmprints)*train_percent)
	train_palmprints, val_palmprints = palmprints[:split1], palmprints[split1:]
	split2 = int(len(val_palmprints)*val_percent)
	val_palmprints, test_palmprints = val_palmprints[:split2], val_palmprints[split2:]

	signatures = list(glob(os.path.join(sign_dir, person, "*").replace("\\", "/")))
	signatures = list(map(lambda x: x.replace("\\", "/"), signatures))
	shuffle(signatures)
	split1 = int(len(signatures)*train_percent)
	train_signatures, val_signatures = signatures[:split1], signatures[split1:]
	split2 = int(len(val_signatures)*val_percent)
	val_signatures, test_signatures = val_signatures[:split2], val_signatures[split2:]

	audios = list(glob(os.path.join(audio_dir, person, "*")))
	audios = list(map(lambda x: x.replace("\\", "/"), audios))
	shuffle(audios)
	split1 = int(len(audios)*train_percent)
	train_audios, val_audios = audios[:split1], audios[split1:]
	split2 = int(len(val_audios)*val_percent)
	val_audios, test_audios = val_audios[:split2], val_audios[split2:]

	train_row = [train_faces, train_palmprints, train_signatures, train_audios]
	data = list(itertools.product(*train_row))
	data = np.array(data)
	labels = np.array([[person]*len(data)]).T
	data = np.append(data, labels, axis=1)
	train_rows.extend(data)

	val_row = [val_faces, val_palmprints, val_signatures, val_audios]
	data = list(itertools.product(*val_row))
	data = np.array(data)
	labels = np.array([[person]*len(data)]).T
	data = np.append(data, labels, axis=1)
	val_rows.extend(data)

	test_row = [test_faces, test_palmprints, test_signatures, test_audios]
	data = list(itertools.product(*test_row))
	if data:
		data = np.array(data)
		labels = np.array([[person]*len(data)]).T
		data = np.append(data, labels, axis=1)
		test_rows.extend(data)


df = pd.DataFrame(train_rows, columns=['face', 'palm_print', 'signature', 'audio', 'label']).sample(frac=1)
df.to_csv(os.path.join(data_dir, "train.csv"))

df = pd.DataFrame(val_rows, columns=['face', 'palm_print', 'signature', 'audio', 'label']).sample(frac=1)
df.to_csv(os.path.join(data_dir, "val.csv"))

df = pd.DataFrame(test_rows, columns=['face', 'palm_print', 'signature', 'audio', 'label']).sample(frac=1)
df.to_csv(os.path.join(data_dir, "test.csv"))
