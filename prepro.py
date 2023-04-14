from hdf_to_df import *
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

def prepro(filename, set_type):

    theSet = hdf_to_df(filename, set_type)
    # Note: Unsure what the value of the window size should be, we will have to do a plot and find the ideal
    # amount of smoothing
    window_size = 61
    norm_X = []
    labels = []
    for i in range(len(theSet)):
        df = theSet[i].rolling(window_size).mean()
        labels.append(df.iloc[61, -1])
        norm_X.append(df)

    return norm_X, labels

