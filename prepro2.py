import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Hi Class,

# A few of you have asked about outlier removal vs noise removal (moving average). 
# Please note that initially, I intended to have separate sections in the course about 
# these two topics, but then opted to simplify things and just stick to noise reduction 
# (moving average). So for the project, please treat both these concepts as the same thing 
# and simply apply the moving average filter to reduce the noise.

# Best,
# Ali

def prepro2(filename):

    theSet = pd.read_csv(filename)
    windowSize = 500
    data = theSet
    data_prior = data.astype('float64')
    data = []
    for i in range(0, len(data_prior), windowSize):
        data_window = data_prior.iloc[i:i + windowSize]
        data.append(data_window)

    if len(data[-1]) < windowSize:
        data.pop()

    theSet = data
    window_size = 61
    norm_X = []
    labels = []
    for i in range(len(theSet)):
        df = theSet[i].rolling(window_size).mean()
        norm_X.append(df)

    return norm_X


