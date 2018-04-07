import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
    return vgg16.VGG16(input_shape=input_shape, include_top=False)

def resnet(input_shape=(224,224,3)):
    return resnet50.ResNet50(input_shape=input_shape, include_top=False)

def inceptionv3(input_shape=(224,224,3)):
    return inception_v3.InceptionV3(input_shape=input_shape, include_top=False)

def inception_res(input_shape=(224,224,3)):
    return inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, include_top=False)

def top_model(input_shape, verbose=False):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_shape))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(50, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))
    if verbose: top_model.summary()
    return top_model

def all_nets():
    return {'simple_cnn':simple_cnn, 'vggnet':vggnet,
           'resnet':resnet, 'inceptionv3':inceptionv3,
           'inception_res':inception_res}