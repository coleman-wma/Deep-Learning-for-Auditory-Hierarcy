#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:54:26 2019

@author: billcoleman

CNN for Auditory Hierarchy
Initial Code based on:
https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529

Local: /Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/EXP_4_CNN10k_DELTA_testSplits_slurm.py

Implemented to train and test the CNN on 100k instances on the same splits used
on SVM for comparative purposes.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

import numpy as np
import pandas as pd

# file management
import pickle
# execution time
import sys
import timeit

# for making directories
import os
from os import path

# train/test splits
from sklearn.model_selection import train_test_split

start_time = timeit.default_timer()

print("This is a script to run CNN models!")

'''
For 100_000 instances - includes augmented data
'''
# load data (flattened and scaled)
# local location: '/Volumes/COLESLAW_1TB/scaled_data/data_cnn_ALLbatches_zOrder.data'
# Kevin Street: '/data/d15126149/datasets/data_cnn_ALLbatches_zOrder.data'
with open('/data/d15126149/datasets/data_cnn_ALLbatches_zOrder.data', 'rb') as filehandle:  
        # read the data as binary data stream
        CNN_tensor = pickle.load(filehandle)

# Testing by augmentation batch
# CNN_tensor = CNN_tensor[:,:,:,0]

print("Shape of input data, CNN_tensor, is:", CNN_tensor.shape)

# labels
# local location: '/Volumes/COLESLAW_1TB/scaled_data/all_labels_100k.data'
# Kevin Street: '/data/d15126149/datasets/all_labels_100k.data'
with open('/data/d15126149/datasets/all_labels_100k.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        all_labels = pickle.load(filehandle)
        
sort_all_labels = all_labels.sort_index()

# Take a slice of labels to match data tensor
# all_labels = all_labels[:100]

print("Shape of input labels, all_labels, is:", all_labels.shape)

'''
Use this for optimising models - vary the random_state if desired
Switch to .csv indices for results generation
'''
# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(CNN_tensor,
                                                    sort_all_labels, # all_labels,
                                                    test_size=0.2,
                                                    random_state=3799,  #1735, # 359
                                                    shuffle=True,
                                                    stratify=all_labels)

# =============================================================================
# # comment back in to generate results
# # indices for 3 random stratified test splits
# # local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_train_val_indices.csv'
# # Kevin Street: '/data/d15126149/datasets/svm_100k_train_val_indices.csv'
# train_val_indices = \
# pd.read_csv('/data/d15126149/datasets/svm_100k_train_val_indices.csv',
#             index_col=[0])
# train_val_indices = pd.DataFrame(train_val_indices)
# print("Shape of train_val_indices: ", train_val_indices.shape)
# print("Columns of train_val_indices: ", train_val_indices.columns)
# 
# # local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_test_indices.csv'
# # Kevin Street: '/data/d15126149/datasets/svm_100k_test_indices.csv'
# test_indices = \
# pd.read_csv('/data/d15126149/datasets/svm_100k_test_indices.csv',
#             index_col=[0])
# test_indices = pd.DataFrame(test_indices)
# print("Shape of test_indices: ", test_indices.shape)
# print("Columns of test_indices: ", test_indices.columns)
# 
# # to index to different splits
# splits = ['0', '1', '2']
# 
# this_split = 0
# 
# X_train = CNN_tensor[train_val_indices[splits[this_split]]]
# y_train = all_labels.loc[train_val_indices[splits[this_split]]]
# X_test = CNN_tensor[test_indices[splits[this_split]]]
# y_test = all_labels.loc[test_indices[splits[this_split]]]
# =============================================================================

'''
We will print training sample shape, test sample shape and total number of
classes present. There are 2 classes. For the
sake of example, we will print two example image from training set and test
set.
'''

print('Training data shape : ', X_train.shape, y_train.shape)
print('Testing data shape : ', X_test.shape, y_test.shape)

# =============================================================================
# print('Training data shape_ : ', X_train_.shape, y_train_.shape)
# print('Testing data shape_ : ', X_test_.shape, y_test_.shape)
# =============================================================================

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

'''
Find the shape of input image then reshape it into input format for training
and testing sets. After that change all datatypes into floats.
'''

# reshaping to provide right shape to model
nRows,nCols,nDims = 40, 157, 1
train_data = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
test_data = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# My data is already categorical so probably don't need this
train_labels_one_hot = tf.keras.utils.to_categorical(y_train)
test_labels_one_hot = tf.keras.utils.to_categorical(y_test)
print('Original label 0 : ', y_train.iloc[0])
print('After conversion to categorical ( one-hot ) : ',
      train_labels_one_hot[0])

# Create Model
def createModel():
    
    '''
    Now create our model. We will add up Convo layers followed by pooling layers.
    Then we will connect Dense(FC) layer to predict the classes. Input data fed to
    first Convo layer, output of that Convo layer acts as input for next Convo
    layer and so on. Finally data is fed to FC layer which try to predict the
    correct labels.
    
    Initial architecture based on Chen2019, a CNN which achieved first place in the
    DCASE Acoustic Scene Classification challenge 2019.
    '''

    model = tf.keras.models.Sequential()
    
    # Convolution 1
    # The first layer with 14 filters of window size 5x5
    model.add(tf.keras.layers.Conv2D(12, (5, 5), padding='same', activation='relu',
                     strides=(2,2), input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Convolution 2
    model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Convolution 3
    model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(56, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Convolution 4
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     strides=(1,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Pooling
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.5)) # added in for 210
    model.add(tf.keras.layers.Dense(nClasses, activation='sigmoid'))
    
    return model

'''
Initialize all parameters and compile our model with rmsprops optimizer.
There are many optimizers for example adam, SGD, GradientDescent, Adagrad,
Adadelta and Adamax ,feel free to experiment with it. Here batch is 256 with
50 epochs.
'''

# If required, load a partially trained model
# load json and create model
# =============================================================================
# json_file = open("models/model_ChenDCASE_10000.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model1 = tf.keras.model_from_json(loaded_model_json)
# # load weights into new model
# model1.tf.keras.load_weights("models/weights.best.hdf5")
# print("Loaded model from disk")
# =============================================================================

# Checking GPU - from Robert Ross
print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))

# To print diagnostics in slurm output
# tf.debugging.set_log_device_placement(True)

model1 = createModel()
batch_size = 128
lr = 0.01
epochs = 1

# assign a name to this model - to keep them seperate
name = "CNN100k_zORDER_optimising_no100_"  # + str(this_split)
namepath = name

# create the folder to hold model objects if it doesn't already exist
if not os.path.exists(os.path.join('models', namepath)):
        os.mkdir(os.path.join('models', namepath))

# Declare optimiser - remember 'rmsprop' worked well for 10_000
optimiser = tf.keras.optimizers.Adam(learning_rate=lr,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  amsgrad=False)

# Compile the model
model1.compile(optimizer=optimiser,
               # use 'categorical_crossentropy' for multi-class
               loss='binary_crossentropy',
               metrics=['accuracy'])


'''
Checkpoint
'''

# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
weightspath = os.path.join('models', namepath, namepath + '.best.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(weightspath,
                                                monitor='val_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max',
                                                save_freq='epoch')
callbacks_list = [checkpoint]

'''
model.summary() is used to see all parameters and shapes in each layers in our
models
'''

model1.summary()


'''
After compiling our model, we will train our model by fit() method, then
evaluate it.
'''


mod_history = model1.fit(train_data, 
                     train_labels_one_hot,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(test_data, test_labels_one_hot),
                     callbacks=callbacks_list)

mod_evaluate = model1.evaluate(test_data, test_labels_one_hot, verbose=2)

# save objects
with open(os.path.join('models', namepath,
                       namepath + '_hist.data'), 'wb') as file_hi:
    pickle.dump(mod_history.history, file_hi)

print("History object saved")

# serialize model to JSON
model_json = model1.to_json()
with open(os.path.join('models', namepath,
                       namepath + '_model.json'), 'w') as json_file:
    json_file.write(model_json)

print("Model saved to json")

print("Loading best model weights from training run, to evaluate...")

# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# just want to seperate these to load best weights to best_model
best_model = model1
best_model.load_weights(weightspath)
print("Loaded model from disk")
 
# evaluate loaded model on test data
best_model.compile(optimizer=optimiser,
               loss='binary_crossentropy',
               metrics=['accuracy'])
score = best_model.evaluate(test_data, test_labels_one_hot, verbose=2)
print("Best Validation %s: %.2f%%" % (best_model.metrics_names[1],
                                      score[1]*100))

with open(os.path.join('models', namepath,
                       namepath + '_eval.data'), 'wb') as file_ev:
    pickle.dump(score, file_ev)

print("Evaluate object saved")


# predict using model and measure precision, recall etc...
y_pred = model1.predict(test_data, batch_size=64, verbose=2)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

# Confusion Matrix
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred_bool)))

bal_acc = balanced_accuracy_score(y_test, y_pred_bool)

print("------------------------------------------------------------------")
print('Balanced accuracy on validation set (y_true Vs y_pred): %.2f%%' % (bal_acc * 100))
print("------------------------------------------------------------------")

print("This model is: ", namepath)

# Export true and predicted labels for Mcnemar
y_pred = pd.DataFrame(y_pred, index=y_test.index)
y_pred_bool_ = pd.Series(y_pred_bool, index=y_test.index)
truePred = pd.DataFrame(pd.concat([y_test, y_pred, y_pred_bool_],
                              axis=1,
                              join='outer'))
truePred.columns=['true', 'pred0', 'pred1', 'pred_bool']
truePred.to_csv(os.path.join('models', namepath,
                       namepath + '_truePred.csv'))

print("Predicted Labels Saved")

# Track the time it took to run the script
stop_time = timeit.default_timer()
total_time = stop_time - start_time

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("(print)Total running time: %d:%d:%d.\n" % (hours, mins, secs))
