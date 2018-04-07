import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications import resnet50, inception_v3, vgg16, inception_resnet_v2

def simple_cnn(num_classes=2, input_shape=(224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def vggnet(num_classes=2, input_shape=(224,224,3)):
    return vgg16.VGG16(classes=num_classes, input_shape=input_shape)

def resnet(num_classes=2, input_shape=(224,224,3)):
    return resnet50.ResNet50(classes=num_classes, input_shape=input_shape)

def inceptionv3(num_classes=2, input_shape=(224,224,3)):
    return inception_v3.InceptionV3(input_shape=input_shape, classes=num_classes)

def inception_res(num_classes=2, input_shape=(224,224,3)):
    return inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, classes=num_classes)

def all_nets():
    return {'simple_cnn':simple_cnn, 'vggnet':vggnet,
           'resnet':resnet, 'inceptionv3':inceptionv3,
           'inception_res':inception_res}