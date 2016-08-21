# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:43:41 2016

@author: nirjhar.sarkar
"""

from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import json



''' A RESFTFul service, using FLASK, to show how to predict image class based on an already saved model'''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
model_location="3class_model_wt.h5"

@app.route('/upload')
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      ret_msg=predict_image_class(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return ret_msg
   else:
        return render_template('upload.html')


''' Load the model and the weights. Need to make this singleton so that the model is loaded only once during start up. Had to move the import inside as Tensorflow was getting loaded twice and was throwing an          error Tensor("cond/pred_id:0", dtype=bool) must be from the same graph as Tensor '''
def load_model(model_location, nb_classes=3):
    
    import os
    #Import Keras related libraries
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD,RMSprop,adam
    from keras.utils import np_utils
    
    #import other python related libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    
    from PIL import Image
    from numpy import *
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split
    import shutil

    from sklearn.preprocessing import LabelEncoder    
    # number of channels
    img_channels = 1
    img_rows, img_cols = 256, 256
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    #batch_size to train
    batch_size = 32
    # number of output classes
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.load_weights(model_location)
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    
    return model


''' Method to predict the image class. '''    
def predict_image_class(filename):

    import numpy as np
    from PIL import Image
    from numpy import *
    from sklearn.preprocessing import LabelEncoder
    
    #print(filename)
    img_rows, img_cols = 256, 256

    
    im=Image.open(filename)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(filename, "PNG")
    
    # number of channels
    img_channels = 1
    
    
    number_classes=3
    
    category_list=['BRAINIX', 'INCISIX', 'PHENIX']
    encoder = LabelEncoder()
    encoder.fit(category_list)

    immatrix = array([array(Image.open(filename)).flatten()],'f')
    train_data = [immatrix]  
    X_test = train_data[0]
    #print(X_test.shape[0], 'test samples')
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test /= 255

    model=load_model(model_location,number_classes)
    img_class=model.predict_classes(X_test)
    #print(img_class[0])
    
    #print(encoder.inverse_transform(img_class)[0] + '   ' +str(img_class[0]) )
    data = {}
    data['Predicted Class'] = encoder.inverse_transform(img_class)[0] + '   ' +str(img_class[0])
    
    probabilities_date={}
    img_class_prob=model.predict_proba(X_test).tolist()
    for i in xrange(number_classes):
        #print(str(img_class_prob[0][i])+'  '+encoder.inverse_transform(i))
        probabilities_date[encoder.inverse_transform(i)]=str(img_class_prob[0][i])
    
    data['Probabilities']= probabilities_date    
    json_data = json.dumps(data,indent=4, sort_keys=True)
        #print(json_data)
    
    os.remove(filename)
    return json_data

#Uncomment for unit testing
#imgloc='/home/bigdata/spyworkspace/RESTFulPy/uploads/IM-0001-0158.dcm.png'
#print(predict_image_class(imgloc))


if __name__ == '__main__':
   app.run(debug = True)