import h5py
import pandas as pd
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Get walking data
jacobWalking = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Walking_Data/Jacob/*.csv")
vidWalking = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Walking_Data/Vid/*.csv")
ethanWalking = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Walking_Data/Ethan/*.csv")

# Get jumping data
jacobJumping = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Jumping_Data/Jacob/*.csv")
vidJumping = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Jumping_Data/Vid/*.csv")
ethanJumping = glob.glob(
    "/Users/johndoe/OneDrive - Queen\'s University/Courses/Year 3/ELEC 390/Project/Data/Jumping_Data/Ethan/*.csv")


# label walking and jumping data with either walking or jumping column
def add_activity_label(file_path, activity_label):
    df1 = pd.read_csv(file_path)
    df = df1[:5000]
    df['activity'] = activity_label
    return df


walking_label = 'walking'
jumping_label = 'jumping'

jsWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in jacobWalking])
jsJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in jacobJumping])
vgWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in vidWalking])
vgJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in vidJumping])
epWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in ethanWalking])
epJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in ethanJumping])

dataJumping = pd.concat([jsJumping_labeled, vgJumping_labeled, epJumping_labeled])
# df1 = dataJumping[:5000]
dataWalking = pd.concat([jsWalking_labeled, vgWalking_labeled, epWalking_labeled])
# df2 = dataWalking[:5000]
# data = pd.concat([df1, df2])
jsCombined = pd.concat([jsWalking_labeled, jsJumping_labeled])
vgCombined = pd.concat([vgWalking_labeled, vgJumping_labeled])
epCombined = pd.concat([epWalking_labeled, epJumping_labeled])
data = pd.concat([dataWalking, dataJumping])
# convert labels of walking = 0, jumping = 1
data.loc[data['activity'] == 'walking', 'activity'] = 0
data.loc[data['activity'] == 'jumping', 'activity'] = 1
data = data.astype('float64')
window_size = 500  # 100 Hz, 5 seconds = 500 samples
overlap = 100
rolling_windows = data.rolling(window_size, overlap)
windows_array = np.array(list(rolling_windows), dtype=object)
windows_concatenated = np.concatenate(windows_array, axis=0)
train_windows, test_windows = train_test_split(windows_concatenated, test_size=0.1, random_state=42)

with h5py.File("data.h5", 'w') as hdf:
    groupJacob = hdf.create_group('/Jacob')
    groupEthan = hdf.create_group('/Ethan')
    groupVid = hdf.create_group('/Vid')
    Train = hdf.create_group('dataset/Train')
    Test = hdf.create_group('dataset/Test')
    Train.create_dataset('windows', data=train_windows)
    Test.create_dataset('windows', data=test_windows)
