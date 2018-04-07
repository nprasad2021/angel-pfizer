
from keras.layers import *
from keras.models import Model
import numpy as np
import os
from scipy.misc import imread, imresize
from keras.callbacks import EarlyStopping
from glob import glob
import pickle as pkl

from keras.preprocessing.image import ImageDataGenerator

train_data_path = "./data/spectrograms/train"
val_data_path = "./data/spectrograms/validation"

BATCH_SIZE = 64
NUM_EPOCH = 50


imsize = (240, 320, 3)

if not os.path.exists("../data/pickled.pkl"):
    
    train = []
    train_classes = []
    for img_class in ["sick", "not_sick"]:
        train_files = glob(f"{train_data_path}/{img_class}/*.png")
        for f in train_files:
            img = imread(f)[:, :, :3]
            img = imresize(img, 0.5)
            train.append(img)
            if img_class == "sick":
                train_classes.append(1)
            else:
                train_classes.append(0)

    X_train = np.array(train, dtype = np.int32)
    y_train = np.array(train_classes, dtype = np.bool)

    val = []
    val_classes = []
    for img_class in ["sick", "not_sick"]:
        val_files = glob(f"{val_data_path}/{img_class}/*.png")
        for f in val_files:
            img = imread(f)[:, :, :3]
            img = imresize(img, 0.5)
            val.append(img)
            if img_class == "sick":
                val_classes.append(1)
            else:
                val_classes.append(0)

    X_val = np.array(val, dtype = np.int32)
    y_val = np.array(val_classes, dtype = np.bool)

    pkl.dump([X_train, y_train, X_val, y_val], open("./data/pickled.pkl", "wb"))
else:
    X_train, y_train, X_val, y_val = pkl.load(open("./data/pickled.pkl", "rb"))

es = EarlyStopping(min_delta=0.1, patience = 15)

input = Input(shape = imsize)
x = Conv2D(32, (3, 3), activation = "linear")(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(3,3)(x)
x = Conv2D(32, (3, 3), activation = "linear")(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(3,3)(x)
x = Flatten()(x)

predictions = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = input, outputs = predictions)
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = NUM_EPOCH, callbacks = [es])


