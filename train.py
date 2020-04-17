
from net.segnet import segnet
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from PIL import Image

# 12 class
NCLASSES = 12
HEIGHT = 416
WIDTH = 416
BATCH_SIZE = 4
TRAIN_TXT = "./CamVid/train.txt"
VALID_TXT = "./CamVid/val.txt"

# data
DATA_DIR = "./CamVid"

# train mode
# ALL_NET
# TRANSFER
# FINE_TUNE
(ALL_NET, TRANSFER, FINE_TUNE) = range(3)
TRAIN_MODE = TRANSFER
FREEZE_LAYERS = 13


def loss(y_true, y_pred):
    # multi classify
    crossloss = K.categorical_crossentropy(y_true, y_pred)
    loss = K.sum(crossloss)/(HEIGHT//2)/(WIDTH//2)
    return loss

def process_lines(start, data, batch_size, base_dir=DATA_DIR):
    X = []
    Y = []
    for i in range(batch_size):
        name = data[start + i].replace('\n', '').split(' ')
        img = Image.open(base_dir + '/' + name[0])
        img = img.resize((WIDTH, HEIGHT))
        img = np.array(img)
        img = img/255
        X.append(img)

        img = Image.open(base_dir + '/' + name[1])
        img = img.resize((int(WIDTH/2), int(HEIGHT/2)))
        img = np.array(img)
        seg_labels = np.zeros((int(HEIGHT/2), int(WIDTH/2), NCLASSES))
        for c in range(NCLASSES):
            seg_labels[:, :, c] = (img[:, :] == c).astype(int)
        seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
        Y.append(seg_labels)
    return X, Y

def generate_arrays_from_file(data, batch_size=BATCH_SIZE, steps_per_epoch=1, base_dir=DATA_DIR):
    i = 0
    while True:
        if i == 0:
            np.random.shuffle(data)
        X, Y = process_lines(i*batch_size, data, batch_size, base_dir=base_dir)
        i = (i+1) % steps_per_epoch
        yield (np.array(X), np.array(Y))

if __name__ == "__main__":
    log_dir = "logs/"

    # keras data_format for tf:(batch, height, width, channels)
    model = segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
    model.summary()

    if TRAIN_MODE == TRANSFER:
        for layer in model.layers[:FREEZE_LAYERS]:
            layer.trainable = False

    WEIGHTS_PATH_NO_TOP = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(WEIGHTS_PATH_NO_TOP, by_name=True)

    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint_period = ModelCheckpoint(log_dir + filepath
                                    , monitor="val_loss"
                                    , save_weights_only=True
                                    , save_best_only=True
                                    , period=3)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss"
                                                    , factor=0.5
                                                    , patience=3
                                                    , verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss"
                                                    , min_delta=0
                                                    , patience=10
                                                    , verbose=1)

    model.compile(loss=loss
                            , optimizer = Adam(lr=1e-3)
                            , metrics=["accuracy"])

    lines_train = []
    lines_valid = []
    with open(TRAIN_TXT, "r") as f:
        lines_train = f.readlines()
    with open(VALID_TXT, "r") as f:
        lines_valid = f.readlines()

    num_train = len(lines_train)
    num_valid = len(lines_valid)
    
    model.fit_generator(generate_arrays_from_file(lines_train
                                        , batch_size=BATCH_SIZE
                                        , steps_per_epoch=max(1, num_train//BATCH_SIZE)
                                        , base_dir=DATA_DIR)
                                , steps_per_epoch=max(1, num_train//BATCH_SIZE)
                                , validation_data=generate_arrays_from_file(lines_valid
                                        , batch_size=BATCH_SIZE
                                        , steps_per_epoch=max(1, num_valid//BATCH_SIZE)
                                        , base_dir=DATA_DIR)
                                , validation_steps=max(1, num_valid//BATCH_SIZE)
                                , epochs=50
                                , initial_epoch=0
                                , callbacks=[checkpoint_period, reduce_lr, early_stopping])
    model.save_weights(log_dir+"last1.h5")