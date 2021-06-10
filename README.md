
# personID
Person Identification using Siamese Network


## Download `datasets`
> https://drive.google.com/drive/folders/1pm-jHY0Czy5bU9869StI4sebGCdggLCA
* Download all the folders in the drive to `datasets` directory
* It has been observed that voxceleb1_wavfile has 193 and 149 folders missing. These are needed when we train the fusion model.

## Generate faces
```sh
python prepare_face_dataset.py
```
The script takes `datasets/Face2500` as input, extracts faces using `MTCNN` package and creates `datasets/FaceCropped`


## Generate palmprints
```sh
python prepare_palmprint_dataset.py
```
The script takes `datasets/CASIA-PalmprintV1` as input, extracts palm ROI and creates `datasets/PalmCropped`


## Prepare meta
* Once the datasets are downloaded create train and validation sets for first 300 individuals
```sh
python prepare_meta.py
# Generates datasets/train.csv, datasets/val.csv and datasets/test.csv
```


# Download the models
>
* Download all the folders in the drive to `models` directory


## Train face model
```sh
pip install git+https://github.com/rcmalli/keras-vggface.git

python train_face_siamese.py  # Check the script for command line arguments
# By default at the end of each epoch the training script only gives loss on first 10 folders

# Validation on all the folders
python validate_face_siamese.py -m models/face_epochs1_lr0.0001_batch32
```


## Train palm_print model
```sh
python train_palm_print_siamese.py  # Check the script for command line arguments
# By default at the end of each epoch the training script only gives loss on first 10 folders

# Validation on all the folders
python validate_palm_print_siamese.py -m models/palm_print_epochs1_lr0.0001_batch32
```

## Train audio model
```sh
python train_audio_siamese.py  # Check the script for command line arguments
# By default at the end of each epoch the training script only gives loss on first 10 folders

# Validation on all the folders
python validate_audio_siamese.py -m models/audio_epochs1_lr0.0001_batch32
```

## Train Siamese model with cascade_fusion on face + palm_print + audio
```sh
python cascade_fusion.py  # Check the script for command line arguments
# By default at the end of each epoch the training script only gives loss on first 50 folders
```