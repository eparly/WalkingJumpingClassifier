from prepro import prepro
from prepro2 import prepro2
from featext import featureExtract

# Generate original model:

train_pre = prepro("data.h5", "Train")   # [0] is norm_train_X, [1] is y_train
test_pre = prepro("data.h5", "Test")     # [0] is norm_test_X, [1] is y_test

train_features = featureExtract(train_pre) # feature extraction of training set
test_features = featureExtract(test_pre) # feature extraction of test set



csv_test_pre = prepro2("xxxxx.csv")          # [0] is norm_csvtest_X, [1] is y_csvtest
csv_test_features = featureExtract(csv_test_pre) # feature extraction of csv test set