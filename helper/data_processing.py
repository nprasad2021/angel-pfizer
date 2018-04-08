from keras.preprocessing import image
from keras.utils import multi_gpu_model
from keras.layers import Input
from glob import glob
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.models import Model


# training generator configuration
def get_gen(dataset, batch_size=40, epochs=200, img_dim = (224,224), input_shape=(224,224,3)):
    print(os.getcwd())
    # dimensions of our images.
    img_width, img_height = img_dim

    input_tensor = Input(shape=input_shape)
    
    training_data_dir = './data/' + dataset + '/train/'
    training_datagen = image.ImageDataGenerator(
        rescale=1./255)

    training_generator = training_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # validation generator configuration
    validation_data_dir = './data/' + dataset + '/validation/'

    validation_datagen = image.ImageDataGenerator(
        rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    return training_generator, validation_generator

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

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

def lr_sched(last_lr):
    return 0.99*last_lr


    