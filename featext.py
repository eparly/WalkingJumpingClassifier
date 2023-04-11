from prepro import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.inspection import DecisionBoundaryDisplay 
from sklearn.decomposition import PCA

features = norm_train.DataFrame(columns=['mean', 'median', 'std', 'max', 'variance'])

# window size
window_size = 125

max_index = 1000 # Sub in for real value

features['mean'] = norm_train[max_index, :-1].rolling(window=window_size).mean()
features['median'] = norm_train[max_index, :-1].rolling(window=window_size).median()
features['std'] = norm_train[max_index, :-1].rolling(window=window_size).std()
features['max'] = norm_train[max_index, :-1].rolling(window=window_size).max()
features['variance'] = norm_train[max_index, :-1].rolling(window=window_size).var()
