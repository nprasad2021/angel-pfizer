
import os
import pickle as pkl
from datetime import datetime
from glob import glob
from math import exp

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imread, imresize
from keras.optimizers import SGD, Adam

train_data_path = "./data/melspectrograms/train"
val_data_path = "./data/melspectrograms/validation"

BATCH_SIZE = 8
NUM_EPOCH = 50

class CustomLRScheduler(Callback):

    def __init__(self, schedule, verbose = True):
        super(CustomLRScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        last_lr = K.get_value(self.model.optimizer.lr)
        lr = self.schedule(last_lr)
        if self.verbose:
            print(f"New learning rate is {lr}")
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

def lr_sched(last_lr):

    return 0.99*last_lr

def pull_data(path):

    train = []
    train_classes = []
    for img_class in ["sick", "not_sick"]:
        train_files = glob(f"{path}/{img_class}/*.png")
        for f in train_files:
            img = imread(f, 'L')
            img = imresize(img, size = 0.5)
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

if not os.path.exists("./data/melspectrograms/pickled.pkl"):
    X_train, y_train = pull_data(train_data_path)
    X_train /= 255

    X_val, y_val = pull_data(val_data_path)
    X_val /= 255

    data_mean = np.mean(X_train)
    X_train -= data_mean
    X_val -= data_mean
    pkl.dump([X_train, y_train, X_val, y_val], open("./data/melspectrograms/pickled.pkl", "wb"))
else:
    X_train, y_train, X_val, y_val = pkl.load(open("./data/melspectrograms/pickled.pkl", "rb"))


class_ratio = np.sum(y_train)/len(y_train)

imsize = X_train[0].shape
timestamp = datetime.now().strftime("%d_%H_%M")

es = EarlyStopping(min_delta=0.1, patience = 15, verbose=True)
tb = TensorBoard(f"./logs/{timestamp}",
                histogram_freq=5,
                write_graph=False,
                write_grads=True)

lr = CustomLRScheduler(lr_sched, verbose = 1)

input = Input(shape = imsize)
x = Conv2D(32, (3, 3), activation = "linear")(input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(32, (3, 3), activation = "linear")(x)
x = LeakyReLU()(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(64, (3, 3), activation = "linear")(x)
x = LeakyReLU()(x)
x = MaxPool2D((2,2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(32, activation = "relu")(x)

predictions = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = input, outputs = predictions)
model.summary()

model.compile(optimizer = "sgd", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, 
            batch_size = BATCH_SIZE,
            validation_data = (X_val, y_val),
            epochs = NUM_EPOCH, 
            callbacks = [es, lr])
