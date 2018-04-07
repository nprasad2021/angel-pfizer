
from keras.layers import *
from keras.models import Model
import numpy as np
import os
from scipy.misc import imread, imresize
from keras.callbacks import EarlyStopping
from glob import glob
import pickle as pkl
import joblib

from keras.preprocessing.image import ImageDataGenerator

train_data_path = "./data/melspectrograms/train"
val_data_path = "./data/melspectrograms/validation"

BATCH_SIZE = 4
NUM_EPOCH = 50


imsize = (480, 640, 1)

#if not os.path.exists("./data/melspectrograms/pickled.pkl"):

def pull_data(path):

    train = []
    train_classes = []
    for img_class in ["sick", "not_sick"]:
        train_files = glob(f"{path}/{img_class}/*.png")
        for f in train_files:
            img = imread(f, 'L')
            img = img.astype(np.float32)
            train.append(img)
            if img_class == "sick":
                train_classes.append(1)
            else:
                train_classes.append(0)

    X_train = np.array(train, dtype = np.float32)
    X_train = X_train[:, :, :, np.newaxis]

    y_train = np.array(train_classes, dtype = np.bool)

    return X_train, y_train

X_train, y_train = pull_data(train_data_path)
X_train /= 255

X_val, y_val = pull_data(val_data_path)
X_val /= 255

data_mean = np.mean(X_train)
X_train -= data_mean
X_val -= data_mean


es = EarlyStopping(min_delta=0.1, patience = 15)

input = Input(shape = imsize)
x = Conv2D(32, (3, 3), activation = "linear")(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(3,3)(x)
x = Conv2D(32, (3, 3), activation = "linear")(x)
x = Activation('relu')(x)
x = MaxPool2D(3,3)(x)
x = Flatten()(x)

predictions = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = input, outputs = predictions)
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, 
            batch_size = BATCH_SIZE,
            validation_data = (X_val, y_val), 
            epochs = NUM_EPOCH, 
            callbacks = [es])


