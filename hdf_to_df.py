import h5py
import pandas as pd
#to use this function in your python scripts, use this line of code
#from hdf_to_df import hdf_to_df
#then call the function with the following line of code
#df = hdf_to_df(filename, type)


#give the filename and the type of data you want to extract as 'Train' or 'Test'
#return a pandas dataframe
def hdf_to_df(filename, type):
    f = h5py.File(filename, "r")

    #testdata contains windows of acceleration samples
    #The last column is the label: 1 for jumping, 0 for walking
    testData = f['dataset'][type]['windows']
    X = testData[:, 0:-1]
    Y = testData[:, -1]
    
    #create a pandas dataframe
    df = pd.DataFrame(X)
    df['label'] = Y
    return df

