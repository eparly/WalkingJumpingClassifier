import pandas as pd

def featureExtract(norm_train):

    features1 = pd.DataFrame(columns=['x_mean', 'x_std', 'x_variance'])
    # window size
    window_size = 10
    overlap = 100
    for i in range(len(norm_train)):
        df = pd.DataFrame(norm_train[i])
        df = df[60:]
        features = pd.DataFrame()
        features.at[i, 'x_mean'] = df.iloc[:, 1].mean()
        features.at[i, 'x_std'] = df.iloc[:, 1].std()
        features.at[i, 'x_variance'] = df.iloc[:, 1].var()
        features.at[i, 'x_kurtosis'] = df.iloc[:, 1].kurt()
        features.at[i, 'x_skew'] = df.iloc[:, 1].skew()
        features.at[i, 'y_mean'] = df.iloc[:, 2].mean()
        features.at[i,'y_std'] = df.iloc[:, 2].std()
        features.at[i, 'y_variance'] = df.iloc[:, 2].var()
        features.at[i, 'y_kurtosis'] = df.iloc[:, 2].kurt()
        features.at[i, 'y_skew'] = df.iloc[:, 2].skew()
        features.at[i, 'z_mean'] = df.iloc[:, 3].mean()
        features.at[i, 'z_std'] = df.iloc[:, 3].std()
        features.at[i, 'z_variance'] = df.iloc[:, 3].var()
        features.at[i,'z_kurtosis'] = df.iloc[:, 3].kurt()
        features.at[i, 'z_skew'] = df.iloc[:, 3].skew()

        features = features.fillna(0)[:]
        features1 = features1.append(features)

    return features1