
# personID
Person Identification using Siamese Network


## Download `datasets`
> https://drive.google.com/drive/folders/1pm-jHY0Czy5bU9869StI4sebGCdggLCA
* Download all the folders in the drive to `datasets` directory
* It has been observed that voxceleb1_wavfile has 193 and 149 folders are missing. These are needed when we train the fusion model.


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

