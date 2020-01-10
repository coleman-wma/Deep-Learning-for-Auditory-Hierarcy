#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:42:04 2019
To score best models on 100k instances with SVM

@author: billcoleman
"""

import pickle
import pandas as pd

# SVM model
from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

'''
LOAD DATA
'''

# load lpms data (flattened and scaled)
# local location: '/Volumes/COLESLAW_1TB/scaled_data/data_svm_ALLbatches_zOrder.data'
# Kevin Street: '/data/d15126149/datasets/data_svm_ALLbatches_zOrder.data'
# local all: '/Volumes/COLESLAW_1TB/scaled_data/data_svm_ALLbatches.data'
# Kevin Street all: '/data/d15126149/datasets/data_svm_ALLbatches.data'
with open('/Volumes/COLESLAW_1TB/scaled_data/data_svm_sc_b1_nonAug.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        allData = pickle.load(filehandle) # was lpms_flat_sc

allData = pd.DataFrame(allData)
allData = allData.iloc[:,0:100]

print("Shape of allData: ", allData.shape)

'''
labels
'''

# local location: '/Volumes/COLESLAW_1TB/scaled_data/all_labels_100k.data'
# Kevin Street: '/data/d15126149/datasets/all_labels_100k.data'

with open('/Volumes/COLESLAW_1TB/scaled_data/all_labels.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        all_labels = pickle.load(filehandle)

print("Shape of all_labels: ", all_labels.shape)

'''
indices for 3 random stratified test splits - train/validation and test
'''

# indices for 3 random stratified test splits
# local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_train_val_indices.csv'
# Kevin Street: '/data/d15126149/datasets/svm_100k_train_val_indices.csv'
train_val_indices = \
pd.read_csv('/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_train_val_indices.csv',
            index_col=[0])
train_val_indices = pd.DataFrame(train_val_indices)
print("Shape of train_val_indices: ", train_val_indices.shape)
print("Columns of train_val_indices: ", train_val_indices.columns)

# local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_test_indices.csv'
# Kevin Street: '/data/d15126149/datasets/svm_100k_test_indices.csv'
test_indices = \
pd.read_csv('/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_100k_test_indices.csv',
            index_col=[0])
test_indices = pd.DataFrame(test_indices)
print("Shape of test_indices: ", test_indices.shape)
print("Columns of test_indices: ", test_indices.columns)


# Declare function to build models
def fit_clf_report(ker, C_val, gamma_val, degree_val):
    
    '''
    Takes parameters and fits a model, then provides scores
    '''
    
    # Model derived from fitting on LPMS Scaled data
    clf = SVC(kernel=ker,
              C=C_val,
              gamma=gamma_val,
              degree=degree_val, # ignored by all kernels except poly
              class_weight='balanced')

    print("------------------------------------------------------------------")
    print('kernel: ', ker, 'C: ', C_val, 'gamma: ', gamma_val, 'degree: ', degree_val)
    print("------------------------------------------------------------------")
    
    # Fit model
    clf.fit(X_train, y_train)

    # BALANCED accuracy scores
    y_true, y_pred = y_test, clf.predict(X_test)
    margin = pd.Series(clf.decision_function(X_test), index=y_true.index)    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    y_pred = pd.Series(y_pred, index=y_true.index)
    print("------------------------------------------------------------------")
    print('Balanced accuracy on test set (y_true Vs y_pred): ', bal_acc)
    print("------------------------------------------------------------------")
    
    ################ CLASS REPORT ###############')
    class_report_ = classification_report(y_true, y_pred)
    print(class_report_)
    
    ################ CONFUSION MATRIX ###############')
    conMat = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n{}".format(conMat))
    
    return y_true, y_pred, margin


'''
# Build a model on best parameters for this split and test on held out test set
'''

# SPLIT 0
print("Starting Split 0")
X_train = allData.loc[train_val_indices['0']]
y_train = all_labels.loc[train_val_indices['0']]
X_test = allData.loc[test_indices['0']]
y_test = all_labels.loc[test_indices['0']]

# Run model parameters (kernel, C, gamma, degree)
y_true0, y_pred0, margin0 = fit_clf_report('rbf', 1, 0.1, 3)

model_0 = pd.DataFrame(pd.concat([y_true0, y_pred0, margin0],
                              axis=1,
                              join='outer'))
model_0.columns=['true', 'pred', 'marg']
model_0.to_csv('results/pSearch_svm100k_zOrder_truePredMargin_0.csv')
print("Split 0 Complete")

# SPLIT 1
print("Starting Split 1")
X_train = allData.loc[train_val_indices['1']]
y_train = all_labels.loc[train_val_indices['1']]
X_test = allData.loc[test_indices['1']]
y_test = all_labels.loc[test_indices['1']]

# Run model parameters (kernel, C, gamma, degree)
y_true1, y_pred1, margin1 = fit_clf_report('rbf', 1, 0.1, 3)

model_1 = pd.DataFrame(pd.concat([y_true1, y_pred1, margin1],
                              axis=1,
                              join='outer'))
model_1.columns=['true', 'pred', 'marg']
model_1.to_csv('results/pSearch_svm100k_zOrder_truePredMargin_1.csv')
print("Split 1 Complete")

# SPLIT 2
print("Starting Split 2")
X_train = allData.loc[train_val_indices['2']]
y_train = all_labels.loc[train_val_indices['2']]
X_test = allData.loc[test_indices['2']]
y_test = all_labels.loc[test_indices['2']]

# Run model parameters (kernel, C, gamma, degree)
y_true2, y_pred2, margin2 = fit_clf_report('rbf', 1, 0.1, 3)

model_2 = pd.DataFrame(pd.concat([y_true2, y_pred2, margin2],
                              axis=1,
                              join='outer'))
model_2.columns=['true', 'pred', 'marg']
model_2.to_csv('results/pSearch_svm100k_zOrder_truePredMargin_2.csv')
print("Split 2 Complete")
