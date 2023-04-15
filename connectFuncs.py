from prepro import prepro
from prepro2 import prepro2
from featext import featureExtract
from model import runmodel, predict
import numpy as np
import pandas as pd
# Generate original model:
def trainModel():
    train_pre, labels_train = prepro("data.h5", "Train")   # [0] is norm_train_X, [1] is y_train
    test_pre, labels_test = prepro("data.h5", "Test")     # [0] is norm_test_X, [1] is y_test

    train_features = featureExtract(train_pre) # feature extraction of training set
    test_features = featureExtract(test_pre) # feature extraction of test set

    runmodel(train_features, labels_train, test_features, labels_test)


def guiFile(filename):
    # [0] is norm_csvtest_X, [1] is y_csvtest
    csv_test_pre = prepro2(filename)
    csv_test_features = featureExtract(csv_test_pre)    # feature extraction of csv test set

    # runmodel(train_features, train_pre[1], csv_test_features, csv_test_pre[1])
    predictions = predict(csv_test_features)

    df = pd.read_csv(filename)
    new_column = np.repeat(predictions, 500)
    new = pd.concat([df, pd.Series(new_column, name='Predictions')], axis=1)
    new.to_csv('predictedData.csv', index=False)
    print(predictions)
    return new_column, predictions

# guiFile("/Users/johndoe/Library/CloudStorage/OneDrive-Queen'sUniversity/Courses/Year 3/ELEC 390/Project/PyCharm/Data/Walking_Data/Jacob/jdata1.csv")