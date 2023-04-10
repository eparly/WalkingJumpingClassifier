import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import h5py

filename = "data.h5"
f = h5py.File(filename, "r")

#testdata contains windows of acceleration samples
#The last column is the label: 1 for jumping, 0 for walking
testData = f['dataset']['Test']['windows']
X = testData[:, 0:-1]
Y = testData[:, -1]

#plot the 4th column of each window, with the label as the color
#--------------------------------------------------------------
def scatterPlot(X, Y):
    fig, ax = plt.subplots()
    for i in range(0, 500):
        
        if testData[i][5] == 1:
            ax.scatter(testData[i][4], testData[i][3], c='red')
        else:
            ax.scatter(testData[i][4], testData[i][3], c='blue')
        

    plt.show()
#--------------------------------------------------------------


#plotting the dataset with the labels as the color using PCA
#--------------------------------------------------------------
def pcaScatter(X, Y):
    X = testData[:, 0:-1]
    Y = testData[:, -1]

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    fig, ax = plt.subplots()
    for i in range(0, 4000):
            
            if Y[i] == 1:
                ax.scatter(X[i][0], X[i][1], c='red')
            else:
                ax.scatter(X[i][0], X[i][1], c='blue')
    #add legend where red = jumping, blue = walking
    plt.legend(['Jumping', 'Walking'])
    #add labels for pca components
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
#--------------------------------------------------------------
#Line plot: You can create a line plot of the X, Y, Z, and absolute acceleration data over time, 
# with different colors for the walking and jumping segments.
# Extract the first 100 trials of absolute acceleration data
def lineplot(X, Y):
    abs_acc = X[:100, 3]  # assuming each trial is 5 seconds long

    # Extract the first 100 labels
    labels = Y[:100]

    # Create a list of segment start times for the first 100 trials
    seg_start_times = [i for i in range(100)]

    # Create a list of colors for each segment based on the label
    colors = ['r' if label == 1 else 'b' for label in labels]

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(seg_start_times, abs_acc, color='k', linewidth=0.5)
    ax.scatter(seg_start_times, abs_acc, c=colors, s=10)

    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Absolute Acceleration')
    ax.set_title('Walking vs Jumping Acceleration Data (First 100 Trials)')

    # Show the plot
    plt.show()
#--------------------------------------------------------------

# Extract the absolute acceleration data and labels

#plotting the z axis produces really good results
def histograms(X, Y, axis):
    if(axis == 'x'):
        axis_num = 0
    elif(axis == 'y'):
        axis_num = 1
    elif(axis == 'z'):
        axis_num = 2
    else:
        axis_num = 3

    abs_acc = X[:, axis_num]
    labels = Y

    # Define the bin size and range for the histograms
    bin_size = 0.15
    bin_range = (-10, 10)

    # Create two sets of absolute acceleration data: one for walking and one for jumping
    walking_data = abs_acc[labels == 0]
    jumping_data = abs_acc[labels == 1]

    # Create the histograms
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax1.hist(walking_data, bins=np.arange(*bin_range, bin_size),
            color='b', alpha=0.5, label='Walking')
    ax2.hist(jumping_data, bins=np.arange(*bin_range, bin_size),
            color='r', alpha=0.5, label='Jumping')

    # Add labels and title
    ax1.set_xlabel(axis + ' Acceleration')
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram of Walking Acceleration Data')
    ax1.legend()

    ax2.set_xlabel(axis +' Acceleration')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of Jumping Acceleration Data')
    ax2.legend()

    # Show the plot
    plt.show()


a=0
# scatterPlot(X, Y)
pcaScatter(X, Y)
# lineplot(X, Y)
# histograms(X, Y, 'z')


