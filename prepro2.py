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

    # Note: Unsure what the value of the window size should be, we will have to do a plot and find the ideal
    # amount of smoothing
    window_size = 50
    sma = theSet.rolling(window_size).mean()

    # Normalizing Data

    X = sma

    norm_X = StandardScaler().fit_transform(X)



    return norm_X

