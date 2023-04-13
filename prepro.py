from hdf_to_df import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
# from sklearn.inspection import DecisionBoundaryDisplay 
from sklearn.decomposition import PCA

# Hi Class,

# A few of you have asked about outlier removal vs noise removal (moving average). 
# Please note that initially, I intended to have separate sections in the course about 
# these two topics, but then opted to simplify things and just stick to noise reduction 
# (moving average). So for the project, please treat both these concepts as the same thing 
# and simply apply the moving average filter to reduce the noise.

# Best,
# Ali

def prepro(filename):

    training_set = hdf_to_df(filename, "Train") 
    test_set = hdf_to_df(filename, "Test")

    # Note: Unsure what the value of the window size should be, we will have to do a plot and find the ideal
    # amount of smoothing
    window_size = 50
    train_sma = training_set.rolling(window_size).mean()
    test_sma = test_set.rolling(window_size).mean()

    # Normalizing Data

    X_train = train_sma.drop('label', axis=1)
    X_test = test_sma.drop('label', axis=1)

    norm_train = StandardScaler().fit_transform(X_train)     # normalize training data
    norm_test = StandardScaler().fit_transform(X_test)       # normalize test data

    y_train = train_sma['label']   # labels for training set
    y_test = test_sma['label']     # labels for test set

    return norm_train, norm_test, y_train, y_test


