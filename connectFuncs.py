from prepro import prepro

# Generate original model:

trainpre = prepro("data.h5", "Train")
testpre = prepro("data.h5", "Test")

csvtest = prepro("xxxxx.csv")
