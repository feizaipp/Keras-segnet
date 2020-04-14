
import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import UpSampling2D, Reshape, Softmax
from keras.models import Model, Input

DATA_FORMAT = "channels_last"

def encoder(input_height=360, input_width=480):
    data_input = Input(shape=(input_height, input_width, 3))

    o = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(data_input)
    o = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(o)
    feature1 = o

    o = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(o)
    o = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(o)
    featrue2 = o

    o = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(o)
    o = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(o)
    featrue3 = o

    o = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(o)
    o = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(o)
    featrue4 = o

    o = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(o)
    o = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(o)
    feature5 = o
    
    return data_input, [feature1, featrue2, featrue3, featrue4, feature5]

def decoder(feature, n_classes, n_up=3):
    assert n_up >= 2
    o = feature

    o = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(o)
    o = Conv2D(512, (3, 3), padding="valid", data_format=DATA_FORMAT)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D(size=(2, 2), data_format=DATA_FORMAT)(o)

    o = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(o)
    o = Conv2D(255, (3, 3), padding="valid", data_format=DATA_FORMAT)(o)
    o = BatchNormalization(o)

    for _ in range(n_up - 2):
        o = UpSampling2D(size=(2, 2), data_format=DATA_FORMAT)(o)
        o = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(o)
        o = Conv2D(128, (3, 3), padding="valid", data_format=DATA_FORMAT)(o)
        o = BatchNormalization()(o)

    o = UpSampling2D(size=(2, 2), data_format=DATA_FORMAT)(o)
    o = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(o)
    o = Conv2D(64, (3, 3), padding="valid", data_format=DATA_FORMAT)(o)
    o = BatchNormalization()(o)

    o = Conv2D(n_classes, (3, 3), padding="same", data_format=DATA_FORMAT)(o)
    return o

def segnet(n_classes, input_height=360, input_width=480, encoder_level = 3):
    data_input, levels = encoder(input_height=input_height, input_width=input_width)

    feature = levels[encoder_level]

    o = decoder(feature, n_classes, n_up=3)

    o = Reshape(((int(input_height/2) * int(input_width/2)), -1))(o)

    o = Softmax()(o)
    model = Model(data_input, o)

    return model
