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

features = norm_train.DataFrame(columns=['mean', 'median', 'std', 'max', 'min', 'range', 'variance', 'kurtosis', 'skew', 'mode'])

# window size
window_size = 125

features['mean'] = norm_train[1, : 4].rolling(window=window_size).mean()
features['median'] = norm_train[1, : 4].rolling(window=window_size).median()
features['std'] = norm_train[1, : 4].rolling(window=window_size).std()
features['max'] = norm_train[1, : 4].rolling(window=window_size).max()
features['min'] = norm_train[1, : 4].rolling(window=window_size).min()
features['range'] = norm_train[1, : 4].rolling(window=window_size).range()
features['variance'] = norm_train[1, : 4].rolling(window=window_size).var()
features['kurtosis'] = norm_train[1, : 4].rolling(window=window_size).kurt()
features['skew'] = norm_train[1, : 4].rolling(window=window_size).skew()
features['mode'] = norm_train[1, : 4].rolling(window=window_size).mode()

features = features.dropna()

