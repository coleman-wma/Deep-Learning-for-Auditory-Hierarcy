#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:14:09 2019

@author: billcoleman
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, ParameterGrid, GridSearchCV

from sklearn.svm import SVC # SVM model

from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# execution time
import sys
import timeit

start_time = timeit.default_timer()

'''
PSEUDOCODE:
    Load in svm zero order data
    Load in different labels (ALL_man_labelled)
    Declare scorers - include precision and recall for nonFG
    Declare function to run grid search
    For each threshold:
        Execute 3 x random stratified test splits
        Run grid search in each
        Generate DF with results of grid search per split
'''

# load lpms data (flattened and scaled)
# local location: '/Volumes/COLESLAW_1TB/ESC/LPMS_flat_scaled_EGAL_data_10000.data'
# Kevin Street: '/data/d15126149/datasets/LPMS_flat_scaled_EGAL_data_10000.data'
with open('/Volumes/COLESLAW_1TB/ESC/LPMS_flat_scaled_EGAL_data_10000.data', 'rb') as filehandle:  
        # read the data as binary data stream
        allData = pickle.load(filehandle) # was lpms_flat_sc

allData = pd.DataFrame(allData)

print("Shape of allData: ", allData.shape)

# labels
# local location: '/Volumes/COLESLAW_1TB/scaled_data/all_labels.data'
with open('/data/d15126149/datasets/all_labels.data',
          'rb') as filehandle:  
        # read the data as binary data stream
        all_labels = pickle.load(filehandle)

# all_labels = pd.DataFrame(all_labels)

print("Shape of all_labels: ", all_labels.shape)
# print("Columns of all_labels: ", all_labels.columns)

# indices for 3 random stratified test splits
# local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_10000_train_val_indices.csv'
train_val_indices = \
pd.read_csv('/data/d15126149/datasets/svm_10000_train_val_indices.csv')
train_val_indices = pd.DataFrame(train_val_indices)
print("Shape of train_val_indices: ", train_val_indices.shape)
print("Columns of train_val_indices: ", train_val_indices.columns)

# local location: '/Users/billcoleman/NOTEBOOKS/EXPERIMENT_4/backup/svm_10000_test_indices.csv'
test_indices = \
pd.read_csv('/data/d15126149/datasets/svm_10000_test_indices.csv')
test_indices = pd.DataFrame(test_indices)
print("Shape of test_indices: ", test_indices.shape)
print("Columns of test_indices: ", test_indices.columns)


'''
Storage and scorers
'''

# objects to track scores
modelRows = {} # to track different model parameters and scores
allmodelDF = pd.DataFrame()

# classifier to use in parameter search
svmMod = SVC(class_weight='balanced')

scorers = { # setting up recall and precision as the metrics we'll use
    'precision': make_scorer(precision_score, pos_label=1), #"FG"
    'recall': make_scorer(recall_score, pos_label=1), #"FG"
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'f1': make_scorer(f1_score, pos_label=1) #"FG"
}


# define function for randomised grid search
def do_svm_GridSearch(train_data, train_labels):

    '''
    Function to execute grid search. The train data and labels need to be fed
    to the function indexed from the allData and all_labels objects
    '''
    
    global allmodelDF
    
    X_train = train_data
    y_train = train_labels
    
    y_train = y_train.astype('int')

    # Grid search of parameters, using 5 fold cross validation, 
    clf = GridSearchCV(estimator = svmMod, 
                       param_grid = param_grid,
                       # n_iter=n_runs,
                       cv = 5,
                       n_jobs = -1, # using all cores
                       scoring=scorers, # works
                       refit=False,
                       iid=False,
                       return_train_score=True,
                       verbose = 2)
    
    
    # Fit model
    clf.fit(X_train, y_train)  # .values.ravel())
    
    test_Prec_means = clf.cv_results_['mean_test_precision']
    test_Prec_stds = clf.cv_results_['std_test_precision']
    test_Rec_means = clf.cv_results_['mean_test_recall']
    test_Rec_stds = clf.cv_results_['std_test_recall']
    test_f1s = clf.cv_results_['mean_test_f1']
    test_accs = clf.cv_results_['mean_test_accuracy']
    test_balAccs = clf.cv_results_['mean_test_balanced_accuracy']
    train_accs = clf.cv_results_['mean_train_accuracy']
    train_balAccs = clf.cv_results_['mean_train_balanced_accuracy']
    
    for te_P_m, te_P_s, te_R_m, te_R_s, te_f1, te_ac, te_bAc, tr_ac, tr_bAc, params in zip(test_Prec_means,
                                                                                           test_Prec_stds, 
                                                                                           test_Rec_means, 
                                                                                           test_Rec_stds, 
                                                                                           test_f1s,
                                                                                           test_accs,
                                                                                           test_balAccs,
                                                                                           train_accs,
                                                                                           train_balAccs,
                                                                                           clf.cv_results_['params']):
    
    
        modelRows.update({'Test_Precision': test_Prec_means, 
                          'Test_Prec_STD': test_Prec_stds,
                          'Test_Recall': test_Rec_means,
                          'Test_Rec_STD': test_Rec_stds,
                          'Test_F1_Score': test_f1s,
                          'Test_Accuracy': test_accs,
                          'Test_Bal_Accuracy': test_balAccs,
                          'Train_Accuracy': train_accs,
                          'Train_Bal_Accuracy': train_balAccs,
                          'Params': clf.cv_results_['params']})
    
      
    
    modelDF = pd.DataFrame(modelRows) 
    allmodelDF = pd.concat([allmodelDF, modelDF], axis=0, join='outer')
    
    return allmodelDF




'''
Assign data and labels
'''

splits = ['0', '1', '2']

for i in splits:
    
    print("Beginning split: ", i)

    '''
    Set data and labels
    '''

    train_val_data = allData.loc[train_val_indices[i]]
    train_val_labels = all_labels.loc[train_val_indices[i]]
    
    '''
    Set Grid
    '''
    param_grid = {'kernel': ['linear'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    '''
    Call Function
    '''
    pSearch_svm_10000 = do_svm_GridSearch(train_val_data,
                                           train_val_labels)
    pSearch_svm_10000.to_csv('results/psearch_svm_10000.csv')
    
    print("Linear kernel complete for split: ", i)
    
    '''
    Set Grid
    '''
    param_grid = {'kernel': ['rbf'],
                  'gamma': [1, 'scale', 1e-1, 1e-2],
                  'C': [0.001, 0.01, 0.1, 1]}
    
    '''
    Call Function
    '''
    pSearch_svm_10000 = do_svm_GridSearch(train_val_data,
                                           train_val_labels)
    pSearch_svm_10000.to_csv('results/psearch_svm_10000.csv')
    
    print("RBF kernel complete for split: ", i)
        
    '''
    Set Grid
    '''
    param_grid = {'kernel': ['poly'],
                   'gamma': [1, 'scale', 1e-1, 1e-2],
                   'C': [0.001, 0.01, 0.1, 1],
                   'degree': [3, 4]}
    '''
    Call Function
    '''
    pSearch_svm_10000 = do_svm_GridSearch(train_val_data,
                                           train_val_labels)
    pSearch_svm_10000.to_csv('results/psearch_svm_10000.csv')
    
    print("Poly kernel complete for split: ", i)
    
    now_time = timeit.default_timer()
    split_time = now_time - start_time
    # output running time in a nice format.
    mins, secs = divmod(split_time, 60)
    hours, mins = divmod(mins, 60)
    print("Time for this split from start: %d:%d:%d.\n" % (hours, mins, secs))

  
'''
Export result
'''
pSearch_svm_10000.to_csv('results/psearch_svm_10000.csv')


'''
Timing script
'''
# Track the time it took to run the script
stop_time = timeit.default_timer()
total_time = stop_time - start_time

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
print("(print)Total running time: %d:%d:%d.\n" % (hours, mins, secs))
