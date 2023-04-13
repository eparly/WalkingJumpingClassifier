from prepro import prepro
from prepro2 import prepro2

# Generate original model:

trainpre = prepro("data.h5", "Train")
testpre = prepro("data.h5", "Test")

csvtest = prepro2("xxxxx.csv")
