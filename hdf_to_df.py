import h5py
import pandas as pd
import numpy as np
#to use this function in your python scripts, use this line of code
#from hdf_to_df import hdf_to_df
#then call the function with the following line of code
#df = hdf_to_df(filename, type)


#give the filename and the type of data you want to extract as 'Train' or 'Test'
#return a pandas dataframe
def hdf_to_df(filename, type):
    dataX = []
    dataY = []
    f = h5py.File(filename, "r")
    data = f['dataset'][type]['windows']
    df_list = []
    for i in range(data.shape[0]):
        df = pd.DataFrame(data[i,:,:])
        df_list.append(df)

    #create a pandas dataframe
    # df = pd.DataFrame(dataX)
    # df['label'] = dataY
    return d_list 

hdf_to_df('data.h5', 'Train')