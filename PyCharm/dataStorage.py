import random

import h5py
import pandas as pd
import glob

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


# combine csv files into one large csv
# jsWalking = pd.concat([pd.read_csv(f) for f in jacobWalking])
# jsWalking.to_csv("jsWalking.csv", index=False, encoding='utf-8-sig')
# vgWalking = pd.concat([pd.read_csv(f) for f in vidWalking])
# vgWalking.to_csv("vgWalking.csv", index=False, encoding='utf-8-sig')

# label walking and jumping data with either walking or jumping column
def add_activity_label(file_path, activity_label):
    data = pd.read_csv(file_path)
    data['activity'] = activity_label
    return data


walking_label = 'walking'
jumping_label = 'jumping'

# jsWalking_labeled = pd.concat([add_activity_label(file, walking_label if 'walking' in file else jumping_label) for file in csv_files])
# vgWalking_labled = pd.concat([add_activity_label(file, walking_label if 'walking' in file else jumping_label) for file in csv_files])

jsWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in jacobWalking])
jsJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in jacobJumping])
vgWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in vidWalking])
vgJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in vidJumping])
epWalking_labeled = pd.concat([add_activity_label(file, walking_label) for file in ethanWalking])
epJumping_labeled = pd.concat([add_activity_label(file, jumping_label) for file in ethanJumping])

walkingCombined = pd.concat([jsWalking_labeled, vgWalking_labeled, epWalking_labeled])
jumpingCombined = pd.concat([jsJumping_labeled, vgJumping_labeled, epJumping_labeled])

# cut into 5 second windows and create with labels for train and without for test

# eliminate label when windowing
dataWalking = walkingCombined.iloc[:, 0:-1]
labelsWalking = walkingCombined.iloc[:, -1]
dataJumping = jumpingCombined.iloc[:, 0:-1]
labelsJumping = jumpingCombined.iloc[:, -1]

window_size = 500  # 100 Hz, 5 seconds = 500 samples
windowsWalking = [dataWalking[i:i + window_size] for i in range(0, len(dataWalking), window_size)]
windowsJumping = [dataJumping[i:i + window_size] for i in range(0, len(dataJumping), window_size)]

random.shuffle(windowsWalking)
random.shuffle(windowsJumping)

with h5py.File("data.h5", 'w') as hdf:
    GDSTrainWalk = hdf.create_group('dataset/Train/Walking')
    GDSTrainJump = hdf.create_group('dataset/Train/Jumping')
    GDSTest = hdf.create_group('dataset/Test')

    groupJacob = hdf.create_group('/Jacob')
    groupEthan = hdf.create_group('/Ethan')
    groupVid = hdf.create_group('/Vid')

    # Determine the index ranges for the two groups
    num_windows_W = len(windowsWalking)
    group1_end_index = int(num_windows_W * 0.9)
    group2_start_index = group1_end_index + 1

    # Add the windows to the appropriate groups (for walking data)
    for i, window in enumerate(windowsWalking):
        if i <= group1_end_index:
            GDSTrainWalk.create_dataset(f'window{i}', data=window.to_numpy())
        else:
            GDSTest.create_dataset(f'window{i}', data=window.to_numpy())

    num_windows_J = len(windowsJumping)
    group1_end_index = int(num_windows_W * 0.9)
    group2_start_index = group1_end_index + 1

    # Add the windows to the appropriate groups (for walking data)
    for i, window in enumerate(windowsJumping):
        if i <= group1_end_index:
            GDSTrainJump.create_dataset(f'window{i}', data=window.to_numpy())
        else:
            GDSTest.create_dataset(f'window{i}', data=window.to_numpy())

# for window_start in range(0, len(dataWalking), window_size):
#     window = dataWalking.iloc[window_start:window_start + window_size]
#     with h5py.File('data.h5', 'a') as f:
#         f['dataset/Train/Walking/walkWindow_{}'.format(i)] = window
#     i += 1
#
# i = 0
# for window_start in range(0, len(dataJumping), window_size):
#     window = dataWalking.iloc[window_start:window_start + window_size]
#     with h5py.File('data.h5', 'a') as f:
#         f['dataset/Train/Jumping/jumpWindow_{}'.format(i)] = window
#     i += 1

# add windowed data into train group
# for i, csv_file in enumerate(jacobWalking):
#     df = pd.read_csv(csv_file)
#     groupJacob.create_dataset(f'dataset{i}', data=df)
