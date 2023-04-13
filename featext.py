import pandas as pd

def featureExtract(norm_train):

    features = pd.DataFrame()

    norm_train = pd.DataFrame(norm_train).dropna()
    
    # window size
    window_size = 500
    overlap=100

    features['x_mean'] = norm_train.iloc[:, 1].rolling(window=window_size).mean()
    features['x_std'] = norm_train.iloc[:, 1].rolling(window=window_size).std()
    features['x_variance'] = norm_train.iloc[:, 1].rolling(window=window_size).var()
    features['x_kurtosis'] = norm_train.iloc[:, 1].rolling(window=window_size).kurt()
    features['x_skew'] = norm_train.iloc[:, 1].rolling(window=window_size).skew()
    features['y_mean'] = norm_train.iloc[:, 2].rolling(window=window_size).mean()
    features['y_std'] = norm_train.iloc[:, 2].rolling(window=window_size).std()
    features['y_variance'] = norm_train.iloc[:, 2].rolling(window=window_size).var()
    features['y_kurtosis'] = norm_train.iloc[:, 2].rolling(window=window_size).kurt()
    features['y_skew'] = norm_train.iloc[:, 2].rolling(window=window_size).skew()
    features['z_mean'] = norm_train.iloc[:, 3].rolling(window=window_size).mean()
    features['z_std'] = norm_train.iloc[:, 3].rolling(window=window_size).std()
    features['z_variance'] = norm_train.iloc[:, 3].rolling(window=window_size).var()
    features['z_kurtosis'] = norm_train.iloc[:, 3].rolling(window=window_size).kurt()
    features['z_skew'] = norm_train.iloc[:, 3].rolling(window=window_size).skew()

    features = features.dropna()

    return features