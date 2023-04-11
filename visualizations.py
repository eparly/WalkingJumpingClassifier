import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns

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
        
        if Y[i] == 1:
            #jumping
            ax.scatter(X[i][0], X[i][2], c='red')
        else:
            #walking
            ax.scatter(X[i][0], X[i][2], c='blue')
    #add legend where red = jumping, blue = walking
    plt.legend(['Jumping', 'Walking'])
    #add axis labels
    plt.ylabel('Z Acceleration')
    plt.xlabel('X Acceleration')

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
    for i in range(0, 1000):
            
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

#plot data using TSNE
#--------------------------------------------------------------
def tsnePlot(X, Y):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
    X= tsne.fit_transform(X)
    fig, ax = plt.subplots()
    for i in range(0, 100):
            
            if Y[i] == 1:
                ax.scatter(X[i][0], X[i][1], c='red')
            else:
                ax.scatter(X[i][0], X[i][1], c='blue')
    #add legend where red = jumping, blue = walking
    plt.legend(['Jumping', 'Walking'])
    #add labels for pca components
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.show()

#function to calculate standard deviation of each column, separated by label
def stdDev(X, Y):
    #separate data by label
    walking = X[Y[:] == 0]
    jumping = X[Y[:] == 1]
    #calculate standard deviation of each column
    std_walking = np.std(walking, axis=0)
    std_jumping = np.std(jumping, axis=0)
    #return a list of the standard deviations
    return [std_walking[0], std_walking[1], std_walking[2], std_walking[3], std_jumping[0], std_jumping[1], std_jumping[2], std_jumping[3]]

#function to plot the standard deviation of each column, separated by label
def stdDevPlot(X, Y):
    std_dev = stdDev(X , Y)
    #create a list of the names of the columns
    names = ['x_walking', 'y_walking', 'z_walking', 'abs_walking', 'x_jumping', 'y_jumping', 'z_jumping', 'abs_jumping']
    #create a list of the standard deviations of the walking data
    walking = [std_dev[0], std_dev[1], std_dev[2], std_dev[3]]
    #create a list of the standard deviations of the jumping data
    jumping = [std_dev[4], std_dev[5], std_dev[6], std_dev[7]]
    #create a list of the x locations for the bars
    x = np.arange(len(names))
    #create the plot
    fig, ax = plt.subplots()
    #create the bars for the walking data
    ax.bar(x, std_dev, 0.4)
    #create the bars for the jumping data
    #add labels and title
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation of Walking and Jumping Data')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    #show the plot
    plt.show()
#--------------------------------------------------------------
#plot correlation matrix between all columns, split into walking and jumping
def correlationMatrix(X, Y):
    #separate data by label
    walking = X[Y[:] == 0]
    jumping = X[Y[:] == 1]
    #create a dataframe for the walking data
    df_walking = pd.DataFrame(walking)
    #create a dataframe for the jumping data
    df_jumping = pd.DataFrame(jumping)
    #create a correlation matrix for the walking data
    corr_walking = df_walking.corr()
    #create a correlation matrix for the jumping data
    corr_jumping = df_jumping.corr()
    #create a list of the names of the columns
    names = ['x', 'y', 'z', 'abs']
    #create the plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    #create the heatmap for the walking data
    sns.heatmap(corr_walking, xticklabels=names, yticklabels=names, ax=ax1)
    #create the heatmap for the jumping data
    sns.heatmap(corr_jumping, xticklabels=names, yticklabels=names, ax=ax2)
    #add labels and title
    ax1.set_title('Walking Data Correlation Matrix')
    ax2.set_title('Jumping Data Correlation Matrix')
    #show the plot
    plt.show()



a=0
# scatterPlot(X, Y)
# pcaScatter(X, Y)
# histograms(X, Y, 'z')
# stdDevPlot(X, Y)
correlationMatrix(X, Y)


