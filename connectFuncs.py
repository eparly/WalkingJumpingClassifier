from prepro import prepro
from prepro2 import prepro2
from featext import featureExtract
from model import runmodel

# Generate original model:

train_pre = prepro("data.h5", "Train")   # [0] is norm_train_X, [1] is y_train
test_pre = prepro("data.h5", "Test")     # [0] is norm_test_X, [1] is y_test

train_features = featureExtract(train_pre[0]) # feature extraction of training set
test_features = featureExtract(test_pre[0]) # feature extraction of test set

runmodel(train_features, train_pre[1], test_features, test_pre[1])


csv_test_pre = prepro2("xxxxx.csv")                 # [0] is norm_csvtest_X, [1] is y_csvtest
csv_test_features = featureExtract(csv_test_pre[0])    # feature extraction of csv test set

runmodel(train_features, train_pre[1], csv_test_features, csv_test_pre[1])