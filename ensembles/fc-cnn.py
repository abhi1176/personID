
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

from plain_models import (base_face_model, base_palm_print_model,
                          base_audio_model_cnn, base_signature_model)
from plain_data_generator import PersonIDSequence

face_model = base_face_model(input_shape=(224, 224, 3))
palm_print_model = base_palm_print_model(input_shape=(90,90,1))
audio_model = base_audio_model_cnn(input_shape=(9, 13, 1))
sign_model = base_signature_model(input_shape=(1000, 5))

concat = layers.Concatenate(axis=1)([audio_model.output, sign_model.output])
concat = layers.Concatenate(axis=1)([palm_print_model.output, concat])
concat = layers.Concatenate(axis=1)([face_model.output, concat])

fc1 = layers.Dense(64)(concat)
fc1 = layers.BatchNormalization()(fc1)
fc1 = layers.ReLU()(fc1)

fc2 = layers.Dense(64)(fc1)
fc2 = layers.BatchNormalization()(fc2)
fc2 = layers.ReLU()(fc2)

fc3 = layers.Dense(64)(fc2)
fc3 = layers.BatchNormalization()(fc3)
fc3 = layers.ReLU()(fc3)

fc4 = layers.Dense(64)(fc3)
fc4 = layers.BatchNormalization()(fc4)
fc4 = layers.ReLU(fc4)

cnn = layers.Conv1D(32, 3, activation='relu', padding='same')(fc4)
cnn = layers.MaxPool1D(pool_size=2, strides=2)(cnn)
cnn = layers.Dropout(0.5)(cnn)
cnn = layers.Flatten()(cnn)
cnn = layers.BatchNormalization()(cnn)
cnn = layers.Dense(300, activation='softmax')(cnn)

model = Model(inputs=[face_model.inputs, palm_print_model.inputs,
                      audio_model.inputs, sign_model.inputs], outputs=cnn)
model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy",
              metrics=['accuracy'])

train_ds = PersonIDSequence(csv_file='datasets/train.csv', batch_size=32)
val_ds = PersonIDSequence(csv_file='datasets/val.csv', batch_size=32)

model.fit(train_ds, epochs=2)
