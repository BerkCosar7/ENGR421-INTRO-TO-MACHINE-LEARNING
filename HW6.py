#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats


# In[2]:


group_means = np.array([[+0.0, +5.5],
                        [+0.0, +0.0],
                        [+0.0, -5.5]])

group_covariances = np.array([[[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+3.2, +2.8],
                               [+2.8, +3.2]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])


# In[3]:


# read data into memory
data_set = np.genfromtxt("hw06_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 3


# In[4]:


# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    
    D=X.shape
    
    centroids=np.genfromtxt("hw06_initial_centroids.csv", delimiter = ",")
    
    distances = np.zeros((D[0], K))
    for k in range(K):
        for i in range(D[0]):
            distances[i, k] = np.sqrt(np.sum((X[i] - centroids[k]) ** 2))
   

    assigns=np.argmin(distances,axis=1)
    
    means = np.zeros((K, D[1]))
    covariances = np.zeros((K, D[1], D[1]))
    priors = np.zeros(K)
    
    for k in range(K):
        cluster_points = X[assigns == k]
        cluster_size = len(cluster_points)
        
        if cluster_size > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_covariance = np.cov(cluster_points, rowvar=False)
            cluster_prior = cluster_size / D[0]
            
            means[k] = cluster_mean
            covariances[k] = cluster_covariance
            priors[k] = cluster_prior
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)


# In[5]:


# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    
    max_iterations = 100
    n_samples, n_features = X.shape
    
    for _ in range(max_iterations):
        # E-step
        responsibilities = np.zeros((n_samples, K))
        for k in range(K):
            det = np.linalg.det(covariances[k])
            inv_cov = np.linalg.inv(covariances[k])
            diff = X - means[k]
            exponent = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            responsibilities[:, k] = priors[k] * np.exp(exponent) / (np.sqrt((2 * np.pi) ** n_features * det))
        
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        
        # M-step
        for k in range(K):
            means[k] = np.dot(responsibilities[:, k], X) / np.sum(responsibilities[:, k])
            diff = X - means[k]
            covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / np.sum(responsibilities[:, k])
            priors[k] = np.mean(responsibilities[:, k])
    
    assignments = np.argmax(responsibilities, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)


# In[43]:



# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    # Plot data points
    for k in range(K):
        plt.plot(X[assignments == k, 0], X[assignments == k, 1], ".", markersize=10, color=cluster_colors[k])

   
    x1_grid, x2_grid = np.meshgrid(np.linspace(-8, 8, 1601), np.linspace(-8, 8, 1601))
    intervals = np.dstack((x1_grid, x2_grid))

   
    for k in range(K):
        em_points = stats.multivariate_normal(means[k], covariances[k]).pdf(intervals)
        if k==0:
            plt.contour(x1_grid, x2_grid, em_points, colors=cluster_colors[k],levels=[0.01])
        else:
            plt.contour(x1_grid, x2_grid, em_points, colors=cluster_colors[k],levels=[0.01])

    
    for k in range(K):
        given_points = stats.multivariate_normal(group_means[k], group_covariances[k]).pdf(intervals)
        if k == 0:
            plt.contour(x1_grid, x2_grid, given_points, linestyles='dashed', colors='k', levels=[0.01])
        else:
            plt.contour(x1_grid, x2_grid, given_points, linestyles='dashed', colors='k',levels=[0.01])

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)


# In[ ]:




