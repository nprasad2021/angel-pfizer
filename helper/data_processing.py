from keras.preprocessing import image
from keras.utils import multi_gpu_model
from keras.layers import Input

# training generator configuration
def get_gen_no_transform(dataset, batch_size=40, epochs=200, img_dim = (224,224), input_shape=(224,224,3)):

    # dimensions of our images.
    img_width, img_height = img_dim

    input_tensor = Input(shape=input_shape)
    
    training_data_dir = 'data/' + dataset + '/train/'
    training_datagen = image.ImageDataGenerator(
        rescale=1./255)

    training_generator = training_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)

    # validation generator configuration
    validation_data_dir = './data/' + dataset + '/validation/'

    validation_datagen = image.ImageDataGenerator(
        rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)

    return training_generator, validation_generator