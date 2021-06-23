
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model


def base_face_model(input_shape=(224, 224, 3)):
    # vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # x = layers.Flatten()(vgg16.output)
    # x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    # # x = layers.Dropout(0.5)(x)
    # # x = layers.Dense(1)(x)
    # model = Model(inputs=vgg16.inputs, outputs=x, name='face_model')
    model = load_model('../standalone/facenet_keras.h5')
    for layer in model.layers:
        layer.trainable = False
    return model


def base_palm_print_model(input_shape=(90, 90, 1)):
    # input_ = layers.Input(shape=(90, 90, 1))
    # x = layers.Conv2D(32, kernel_size=3, padding='same')(input_)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)

    # x = layers.Conv2D(32, kernel_size=3, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)

    # x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)

    # model = Model(inputs=input_, outputs=x, name='palm_print_model')
    model = load_model("../standalone/palm_model_e10_lr0.0001.h5")
    model = Model(inputs=model.inputs, outputs=model.get_layer('last_dense').output)
    for layer in model.layers:
        layer._name = "palm_" + layer._name
        layer.trainable = False
    return model


def base_audio_model_cnn(input_shape=(9, 13, 1)):
    # input_ = layers.Input(shape=input_shape)
    # x = layers.Conv2D(16, 3, activation='relu', strides=(1, 1),
    #                   padding='same')(input_)
    # x = layers.Conv2D(32, 3, activation='relu', strides=(1, 1),
    #                   padding='same')(x)
    # x = layers.Conv2D( 8, 3, activation='relu', strides=(1, 1),
    #                   padding='same')(x)
    # x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)

    # model = Model(inputs=input_, outputs=x, name='audio_model_cnn')
    model = load_model("../standalone/audio_model_e80_lr0.001.h5")
    model = Model(inputs=model.inputs, outputs=model.get_layer('last_dense').output)
    for layer in model.layers:
        layer._name = "audio_" + layer._name
        layer.trainable = False
    return model


def base_audio_model_rnn(input_shape=(9, 13)):
    input_ = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(input_)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(16, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(8, activation='relu'))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    model = Model(inputs=input_, outputs=x, name='audio_model_rnn')
    return model


def base_signature_model(input_shape=(1000, 5)):
    input_ = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(input_)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(16, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(8, activation='relu'))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)

    model = Model(inputs=input_, outputs=x, name='audio_model_rnn')
    return model
