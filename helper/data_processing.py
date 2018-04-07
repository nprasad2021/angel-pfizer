from keras.preprocessing import image
from keras.utils import multi_gpu_model
from keras.layers import Input
from glob import glob
import os

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

def collect_data(dataset, ext):

    '''
    train_data_path = "./data/" + dataset + "/train/"
    val_data_path = "./data/" + dataset + "/validation/"

    if not os.path.exists("./data/" + dataset + "/pickled.pkl"):
        
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

        pkl.dump([X_train, y_train, X_val, y_val], open("./data/" + dataset + "/pickled.pkl", "wb"))
    else:
        X_train, y_train, X_val, y_val = pkl.load(open("./data/" + dataset + "/pickled.pkl", "rb"))

    return X_train, y_train, X_val, y_val
    '''
    pass


    