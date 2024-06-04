#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")


# In[3]:


# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]


# In[4]:


# set drawing parameters
minimum_value = 0.0
maximum_value = 2.0
x_interval = np.arange(start = minimum_value, stop = maximum_value + 0.002, step = 0.002)


# In[5]:


def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)


# In[6]:


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    
    num_bins = len(left_borders)
    y_hat = np.zeros(len(x_query))
    
    for i in range(len(x_query)):
        xq = x_query[i]
        
        bin_counts = np.zeros(num_bins)
        bin_sums = np.zeros(num_bins)
        
        
        for j in range(len(x_train)):
            xt = x_train[j]
            
            bin_index = np.searchsorted(right_borders, xt) - 1
            
            bin_sums[bin_index] += y_train[j]
            bin_counts[bin_index] += 1
        
        bin_index = np.searchsorted(right_borders, xq) - 1
        
        if bin_counts[bin_index] > 0:
            y_hat[i] = bin_sums[bin_index] / bin_counts[bin_index]
        else:
            y_hat[i] = np.nan
    # your implementation ends above
    return(y_hat)
    
bin_width = 0.10
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))


# In[7]:


# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    
    y_hat = np.zeros(len(x_query))
    
    for i in range(len(x_query)):
        xq = x_query[i]
        left_bin = xq - 0.5 * bin_width
        right_bin = xq + 0.5 * bin_width
        
        indices = (x_train > left_bin) & (x_train <= right_bin)
        #I believe x_train >=left_bin) & (x_train < right_bin) also correct,
        #because its modeller's choice to break the ties, we can choose both;
        #put the border to right or left. But I checked the given output and
        #the choice I coded matches with the given output therefore I choose that.
        if np.any(indices):
            y_hat[i] = np.mean(y_train[indices])
        else:
            y_hat[i] = np.nan
            
            
    # your implementation ends above
    return(y_hat)

bin_width = 0.10

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))


# In[8]:


# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
        # your implementation starts below
    
    y_hat = np.zeros(len(x_query))
    
    for i in range(len(x_query)):
        xq = x_query[i]
        weights = np.exp(-(x_train - xq)**2 / (2 * bin_width**2)) / (np.sqrt(2 * np.pi) * bin_width)
        y_sum = np.sum(weights * y_train)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            y_hat[i] = y_sum / weight_sum
        else:
            y_hat[i] = np.nan
            
    # your implementation ends above
    return(y_hat)

bin_width = 0.02

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))

