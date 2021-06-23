
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

from plain_models import (base_face_model, base_palm_print_model,
                          base_audio_model_cnn, base_signature_model)
from plain_data_generator import PersonIDSequence

face_model = base_face_model(input_shape=(224, 224, 3))
palm_print_model = base_palm_print_model(input_shape=(90,90,1))
audio_model = base_audio_model_cnn(input_shape=(9, 13, 1))
# sign_model = base_signature_model(input_shape=(1000, 5))

# merge_1 = layers.Concatenate(axis=1)([audio_model.output, sign_model.output])
# merge_1 = layers.Dense(64)(merge_1)
# merge_1 = layers.BatchNormalization()(merge_1)
# merge_1 = layers.ReLU()(merge_1)

merge_1 = audio_model.output

merge_2 = layers.Concatenate(axis=1)([palm_print_model.output, merge_1])
merge_2 = layers.Dense(64)(merge_2)
merge_2 = layers.BatchNormalization()(merge_2)
merge_2 = layers.ReLU()(merge_2)

merge_3 = layers.Concatenate(axis=1)([face_model.output, merge_2])
merge_3 = layers.BatchNormalization()(merge_3)
merge_3 = layers.Dense(300, activation='softmax')(merge_3)

# inputs = [face_model.inputs, palm_print_model.inputs, audio_model.inputs, sign_model.inputs]
inputs = [face_model.inputs, palm_print_model.inputs, audio_model.inputs]

model = Model(inputs=inputs, outputs=merge_3)
model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy",
              metrics=['accuracy'])

# train_ds = PersonIDSequence(csv_file='../datasets/train.csv', batch_size=16)
# val_ds = PersonIDSequence(csv_file='../datasets/val.csv', batch_size=16)

from plain_data_generator_2 import PersonIDSequence
train_ds = tf.data.Dataset.from_generator(
    PersonIDSequence(csv_file='../datasets/train.csv', batch_size=16))
train_ds = train_ds.prefetch(2)

val_ds = tf.data.Dataset.from_generator(
    PersonIDSequence(csv_file='../datasets/val.csv', batch_size=16))
val_ds = val_ds.prefetch(2)

model.fit(train_ds, epochs=2)
