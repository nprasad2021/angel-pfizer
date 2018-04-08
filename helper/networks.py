import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.applications import resnet50, inception_v3, vgg16, inception_resnet_v2

def simple_cnn(input_shape=(224,224,3), freeze=0):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

def vggnet(input_shape=(224,224,3), freeze=0):
    model = vgg16.VGG16(input_shape=input_shape, include_top=False)
    for layer in model.layers[:freeze]:
        layer.trainable = False

def resnet(input_shape=(224,224,3), freeze=0):
    model = resnet50.ResNet50(input_shape=input_shape, include_top=False)
    for layer in model.layers[:freeze]:
        layer.trainable = False

def inceptionv3(input_shape=(224,224,3), freeze=0):
    model = inception_v3.InceptionV3(input_shape=input_shape, include_top=False)
    for layer in model.layers[:freeze]:
        layer.trainable = False

def inception_res(input_shape=(224,224,3), freeze=0):
    model = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, include_top=False)
    for layer in model.layers[:freeze]:
        layer.trainable = False

def top_model(input_shape, freeze=0, verbose=False):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def audio_model(input_shape=(224,224,3), freeze=0):

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4

    model = Sequential()
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('elu'))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    return model

def audio_model_2(input_shape=(224,224,3), freeze=0):

    nb_filters = 64  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4

    model = Sequential()
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('elu'))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    return model

def tim_model(input_shape=(224,224,3), freeze=0):
    input = Input(shape = input_shape)
    x = Conv2D(32, (3, 3), activation = "linear")(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation = "linear",
                kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU()(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(64, (3, 3), activation = "linear",
                kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU()(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(32, activation = "relu")(x)

    predictions = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = input, outputs = predictions)

    return model
def jaron(input_shape=(224,224,3), freeze=0):
    
    model = Sequential()
    # input: 60x41 data frames with 2 channels => (60,41,2) tensors

    # filters of size 3x3 - paper describes using 5x5, but their input data is 128x128
    f_size = 3

    # Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the
    # shape (24,1,f,f).  This is followed by (4,2) max-pooling over the last
    # two dimensions and a ReLU activation function
    model.add(Conv2D(24, (f_size, f_size), padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the 
    # shape (48, 24, f, f). Like L1 this is followed by (4,2) max-pooling 
    # and a ReLU activation function.
    model.add(Conv2D(48, (f_size, f_size), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    # Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the
    # shape (48, 48, f, f). This is followed by a ReLU but no pooling.
    model.add(Conv2D(48, (f_size, f_size), border_mode='valid'))
    model.add(Activation('relu'))

    return model

def all_nets():
    return {'simple_cnn':simple_cnn, 'vggnet':vggnet,
           'resnet':resnet, 'inceptionv3':inceptionv3,
           'inception_res':inception_res, 'audio_model':audio_model,
           'audio_model_2':audio_model_2, 'tim_model':tim_model, 'jaron':jaron}