
import os

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from glob import glob

# create the detector, using default weights
detector = MTCNN()

face_dir = "datasets/Face2500"
output_dir = "datasets/FaceCropped"
shape = (224, 224)

for img_file in glob(face_dir+"/*/*"):
    print("Working on", img_file)
    dirname = os.path.basename(os.path.dirname(img_file))
    filename = os.path.basename(img_file)
    output_file = os.path.join(output_dir, dirname, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    image = Image.open(img_file)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    try:
        x1, y1, width, height = results[0]['box']
    except:
        print("[INFO] Unable to find face in: {}".format(filename))
    else:
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        face = Image.fromarray(face)
        face = face.resize(shape)
        face.save(output_file)
