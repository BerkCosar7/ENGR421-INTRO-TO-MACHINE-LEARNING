import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "hw02_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    
    K=np.max(y)
    class_priors = np.array([np.mean(y == (c + 1)) for c in range(K)])
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    
    K=np.max(y)
    D=len(X[0,:])
    sample_means = np.array([[np.mean(X[:,i][y == (c + 1)]) for i in range(D)] for c in range(K)])
    
    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    
    
    K = np.max(y)
    D = len(X[0,:])
    
    sample_means = estimate_class_means(X, y)
    sample_covariances = np.zeros((K, D, D))
    
    for c in range(K):
        
        index_c = np.where(y == c+1)[0]
        X_c = X[index_c]
        cov_c = np.zeros((D, D))
        
        for i in range(len(index_c)):
            
            A = np.reshape(X_c[i] - sample_means[c], (D, 1)) # without reshape, it is like [3 4] now it is [[3][4]]
                                                             # therefore now i can get both rows correctly.
            cov_c += np.matmul(A,A.T)
            
        cov_c /= len(index_c) 
        
        sample_covariances[c] = cov_c
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    
    
    D = len(X[0,:])
    N = len(X[:,0])
    K = len(class_priors)
    
    score_values=np.zeros((N,K))
    
    for c in range(K):
        inv_cov = np.linalg.inv(class_covariances[c])
        det_cov = np.linalg.det(class_covariances[c])
        for i in range(N):
            
            A = np.reshape(X[i] - class_means[c], (D, 1))
            score_values[i,c] = -0.5*D*np.log(2*math.pi) -0.5*np.log(det_cov) -0.5* np.matmul(np.matmul(A.T,inv_cov),A)+np.log(class_priors[c])
    
    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    K=len(scores[0,:])
    
    y_pred = np.argmax(scores, axis=1) +1
    
    confusion_matrix = pd.crosstab(y_pred.T, y_truth.T).values
    
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below
    
    
    K = np.max(y)
    D = len(X[0,:])
    N=  len(X[:,0])
    sample_covariances=np.zeros((K,D,D))
    sample_means = np.array([np.mean(X[:,i]) for i in range(D)])
    
    
    
    
    for c in range(K):
        cov_c=np.zeros((D,D))
        for i in range(N):
            
            A = np.reshape(X[i] - sample_means, (D, 1)) 
            
            cov_c += np.matmul(A,A.T)
            
           
        
        sample_covariances[c] = cov_c/N
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")
