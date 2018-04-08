### Keras Models
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import *
from keras import metrics, optimizers
from keras.models import model_from_json
import time 
import numpy as np
import os
import csv

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1

def MadMaxPool(team_name='madmaxpool', path_to_data_source='/om/user/nprasad/angel-pfizer/data/continuous_wavelet_transform', path_to_model_weights='/om/user/nprasad/angel-pfizer/models/5/top_long/cwt_images/vggnet.hdf5'):
    # load json file and recreate model

    model = load_model(path_to_model_weights)
    # evaluate loaded model with test data
    testing_data_dir = path_to_data_source + '/test'
    
    testing_datagen = image.ImageDataGenerator(
                    rescale=1./255,
                    featurewise_center=True,
                    featurewise_std_normalization=True)
    
    testing_generator = testing_datagen.flow_from_directory(
        testing_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
        )

    idx = 0
    files_count = testing_generator.n
    predictions = np.zeros((files_count, 2))
    
    while testing_generator.total_batches_seen < files_count:
        if idx % 200 == 0:
            print(idx)
        # calclate time to read data and obtain prediction
        start_time = time.time()
        x_test, y_test = testing_generator.next()
        y_pred = loaded_model.predict(x_test, batch_size=1)
        elapsed_time = time.time() - start_time

        predictions[idx, 0] = np.argmax(y_pred) == np.argmax(y_test)
        predictions[idx,1] = elapsed_time
        idx += 1

    test_acc = round(np.mean(predictions[:,0]), 8)
    avg_time = round(np.mean(predictions[:,1]), 8)
    std_time = round(np.std(predictions[:,1]), 8)

    # path to csv file for performance recordings
    path_to_csv = 'sickness_model_performances.csv'
    colnames = ['team_name', 'model_name', 'test_acc', 'avg_time', 'std_time']
    
    if not os.path.exists(path_to_csv):
        with open(path_to_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(colnames)
    model_name = 'final'
    csv_dict = {'team_name': team_name, 
                  'model_name': model_name, 
                  'test_acc': test_acc, 
                  'avg_time': avg_time, 
                  'std_time': std_time}
    
    print
    print("%s obtained %.3f%% accuracy" % (team_name, 100*test_acc))
    print("average prediction time: %.5f s" % (avg_time))
    print("standard dev prediction time: %.5f s" % (std_time))

    with open(path_to_csv, 'a') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=colnames)
        writer.writerow(csv_dict)

MadMaxPool()