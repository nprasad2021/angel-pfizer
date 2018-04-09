from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from sklearn import preprocessing
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/vggnet.hdf5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
from keras.models import load_model
model = load_model(MODEL_PATH)
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

#image.ImageDataGenerator(rescale=1./255,featurewise_center=True,featurewise_std_normalization=True)
#img_path = '/Users/krish/Documents/Audio_Classification/keras-flask-deploy-webapp-master/uploads/audioset__1keOOsT738_30_35.png'

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    #data_gen = image.ImageDataGenerator(rescale=1./255,featurewise_center=True,featurewise_std_normalization=True)
    #img_1 = data_gen.flow_from_directory('/Users/krish/Documents/Audio_Classification/keras-flask-deploy-webapp-master/uploads')
    # Preprocessing the image
    x = image.img_to_array(img)
    x /= 255
    x -= np.mean(x)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])
        if preds[0]>0.5:
            result = 'Sick'
        else:
            result = 'Not Sick'
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
