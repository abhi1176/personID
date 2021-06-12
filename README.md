
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

## Train cascade fusion model
> Run `cascade_fusion.ipynb` notebook
