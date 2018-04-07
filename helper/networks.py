import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.applications import resnet50, inception_v3, vgg16, inception_resnet_v2

def simple_cnn(input_shape=(224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

def vggnet(input_shape=(224,224,3)):
    return vgg16.VGG16(input_shape=input_shape, include_top=False, weights=None)

def resnet(input_shape=(224,224,3)):
    return resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=None)

def inceptionv3(input_shape=(224,224,3)):
    return inception_v3.InceptionV3(input_shape=input_shape, include_top=False, weights=None)

def inception_res(input_shape=(224,224,3)):
    return inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, include_top=False, weights=None)

def top_model(input_shape, verbose=False):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def audio_model(input_shape=(224,224,3)):

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4

    model = Sequential()
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('elu'))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    return model

def all_nets():
    return {'simple_cnn':simple_cnn, 'vggnet':vggnet,
           'resnet':resnet, 'inceptionv3':inceptionv3,
           'inception_res':inception_res, 'audio_model':audio_model}