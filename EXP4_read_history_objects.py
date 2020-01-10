#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:43:04 2019

@author: billcoleman
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# file management
import pickle

# train/test splits
from sklearn.model_selection import train_test_split

path = '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/models_slurm/'
folder = 'CNN10k_zORDER_FINAL_split_2_VA82_ep73'
filename = 'nonAug_126_as122_rmsRHOp8_DECp8_LR0p01_BA128_ep100_hist.data'

with open(path + folder + '/' + filename,
          'rb') as filehandle:  
        # read the data as binary data stream
        check = pickle.load(filehandle)

'''
Plotting loss and accuracy
See:
https://github.com/dshahid380/Applcation-of-CNN/blob/master/CIFAR_10_using_data_augmentation.ipynb
'''

plt.figure(figsize=(12,6))
plt.plot(check['loss'],'r',linewidth=3.0)
plt.plot(check['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.grid(True)
plt.title('Loss Curves ::: ' + folder, fontsize=10)
plt.xlim((-2,102))
plt.ylim((-0.2,5))
plt.show()


plt.figure(figsize=(12,6))
plt.xlim((-2,102))
plt.ylim((0,1.1))
plt.plot(check['accuracy'],'r',linewidth=3.0)
plt.plot(check['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.grid(True)
plt.title('Accuracy Curves ::: ' + folder, fontsize=10)
plt.show()



'''
Load model and predict on validation data to get average class accuracy
'''
'''
For 10_000 instances - non augmented data
'''
# load data (flattened and scaled)
# local location: '/Volumes/COLESLAW_1TB/scaled_data/data_cnn_sc_b1_nonAug.data'
# Kevin Street: '/data/d15126149/datasets/data_cnn_sc_b1_nonAug.data'
with open('/Volumes/COLESLAW_1TB/scaled_data/data_cnn_sc_b1_nonAug.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        CNN_tensor = pickle.load(filehandle)

#CNN_tensor = CNN_tensor[:10000,:,:,:]

# labels
# local location: '/Volumes/COLESLAW_1TB/scaled_data/all_labels.data'
# Kevin Street: '/data/d15126149/datasets/all_labels.data'
with open('/Volumes/COLESLAW_1TB/scaled_data/all_labels.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        all_labels = pickle.load(filehandle)

# sort series by index so its the same order as the data
all_labels_sort = all_labels.sort_index()

#all_labels_sort = all_labels_sort[:10000]


'''
Load model and weights
'''

json_name = 'nonAug_11Convos_wDO_LR0pt01_ep100_model'
model_name = 'nonAug_11Convos_wDO_LR0pt01_ep100.best.hdf5'

json_file = open(path + folder + '/' + json_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights(path + folder + '/' + model_name)
print("Loaded model from disk")


'''
Replicate same train/test split as used in training
'''

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(CNN_tensor,
                                                    all_labels_sort,
                                                    test_size=0.2,
                                                    random_state=359,
                                                    shuffle=True,
                                                    stratify=all_labels_sort)

print('Training data shape : ', X_train.shape, y_train.shape)
print('Testing data shape : ', X_test.shape, y_test.shape)

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

'''
Find the shape of input image then reshape it into input format for training
and testing sets. After that change all datatypes into floats.
'''

nRows,nCols,nDims = 40, 157, 3
train_data = X_train
test_data = X_test
input_shape = (nRows, nCols, nDims)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# one hot
train_labels_one_hot = tf.keras.utils.to_categorical(y_train)
test_labels_one_hot = tf.keras.utils.to_categorical(y_test)

'''
model.summary() is used to see all parameters and shapes in each layers in our
models
'''

model1.summary()

# predict using model and measure precision, recall etc...
y_pred = model1.predict(X_test, batch_size=64, verbose=2)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

# Confusion Matrix
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred_bool)))

