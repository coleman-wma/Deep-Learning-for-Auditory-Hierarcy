# Experiment-4
Python files written for Experiment 4, which investigated the performance of SVM and CNN models on an auditory hierarchy prediction problem.

#### EXP_4_CNN100k_zORDER_optim_slurm.py
Optimising CNN architecture and parameters. Trains a model for 100 epochs and records results.

#### EXP4_read_history_objects.py
Reads a history object from the training of a CNN model to track metrics over epochs.

#### EXP_4_svm_10000_fit_params_slurm.py
Runs a parameter grid search for the SVM algorithm.

#### EXP_4_SVM100k_zOrder_bestModels_slurm.py
Takes the best parameters found for the SVM algorithm in a grid search and evaluates them on three held-out test set splits.
