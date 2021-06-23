
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

cnn_1 = layers.Conv1D(64, 3, padding='same')(concat)
cnn_1 = layers.BatchNormalization()(cnn_1)
cnn_1 = layers.ReLU(cnn_1)
cnn_1 = layers.MaxPool1D(pool_size=2, strides=2)(cnn_1)

cnn_2 = layers.Conv1D(64, 3, padding='same')(cnn_1)
cnn_2 = layers.BatchNormalization()(cnn_2)
cnn_2 = layers.ReLU(cnn_2)
cnn_2 = layers.MaxPool1D(pool_size=2, strides=2)(cnn_2)

fc = Dense(64, activation="softmax")(cnn_2)

model = Model(inputs=[face_model.inputs, palm_print_model.inputs,
                      audio_model.inputs, sign_model.inputs], outputs=fc)
model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy",
              metrics=['accuracy'])

train_ds = PersonIDSequence(csv_file='datasets/train.csv', batch_size=32)
val_ds = PersonIDSequence(csv_file='datasets/val.csv', batch_size=32)

model.fit(train_ds, epochs=2)
