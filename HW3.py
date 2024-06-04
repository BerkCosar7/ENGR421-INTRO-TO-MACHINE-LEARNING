#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd


# In[2]:


X = np.genfromtxt("hw03_data_points.csv", delimiter = ",")
y = np.genfromtxt("hw03_class_labels.csv", delimiter = ",").astype(int)


# In[3]:


i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(1 - i1, cmap = "gray")
plt.show()
fig.savefig("hw03_images.pdf", bbox_inches = "tight")


# In[4]:



# STEP 3
# first 60000 data points should be included to train
# remaining 10000 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    
    X_train=X[0:60000]
    y_train=y[0:60000]
    X_test=X[60000:70000]
    y_test=y[60000:70000]
    
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[5]:


# STEP 4
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def sigmoid(X, W, w0):
    # your implementation starts below
    
    scores= 1 / (1 + np.exp(-(np.matmul(X, W) + w0)))
    
    # your implementation ends above
    return(scores)


# In[6]:



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def one_hot_encoding(y):
    # your implementation starts below
    
    classes = np.unique(y)
    
    K = len(classes)
    N= len(y)
    
    Y = np.zeros((N, K))
    
    for i in range(N):
        index = np.where(classes == y[i])[0][0]
        Y[i, index] = 1
    
    
    
    # your implementation ends above
    return(Y)


# In[7]:


np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.01, high = 0.01, size = (D, K))
w0_initial = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# In[8]:


# STEP 6
# assuming that there are D features and K classes
# should return a numpy array with shape (D, K)
def gradient_W(X, Y_truth, Y_predicted):
    # your implementation starts below
    
    # 0.5* (Y_truth-Y_predicted)^2 's derivative with respect to W.
    # = (Y_truth-Y_predicted)* (Y_truth-Y_predicted)'s derivative with respect to W
    # = -(Y_truth-Y_predicted)* (Y_predicted's derivative w.r.t W) (Y_predicted = sigmoid (W.T*x+w0))
    # Lets say W.T*x+w0 = a, derivative of sigmoid(a) w.r.t W.
    # derivative of sigmoid(a)=sigmoid(a)*(1-sigmoid(a))
    # chain rule: dsigmoid(a)/da * da/dW
    # = sigmoid(a)*(1-sigmoid(a))*X = Y_predicted*(1-Y_predicted)*X
    #Therefore gradient= -(Y_truth-Y_predicted)*Y_predicted*(1-Y_predicted)*X
    
    gradient = (np.array([-np.matmul((Y_truth[:, i] - Y_predicted[:, i])  
                                       * Y_predicted[:, i] * (1 - Y_predicted[:, i]), X[0:len(Y_train)]) for i in range(K)]).T)
    
    # your implementation ends above
    return(gradient)

# assuming that there are K classes
# should return a numpy array with shape (1, K)
def gradient_w0(Y_truth, Y_predicted):
    # your implementation starts below
    
    # 0.5* (Y_truth-Y_predicted)^2 's derivative with respect to w0.
    # = (Y_truth-Y_predicted)* (Y_truth-Y_predicted)'s derivative with respect to w0
    # = -(Y_truth-Y_predicted)* (Y_predicted's derivative w.r.t w0) (Y_predicted = sigmoid (W.T*x+w0))
    # Lets say W.T*x+w0 = a, derivative of sigmoid(a) w.r.t w0.
    # derivative of sigmoid(a)=sigmoid(a)*(1-sigmoid(a))
    # chain rule: dsigmoid(a)/da * da/dw0
    # = sigmoid(a)*(1-sigmoid(a)) = Y_predicted*(1-Y_predicted)
    #Therefore gradient= -(Y_truth-Y_predicted)*Y_predicted*(1-Y_predicted)
    
    gradient = (-np.sum((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted), axis = 0)) 
    
   
    # your implementation ends above
    return(gradient)


# In[9]:


# STEP 7
# assuming that there are N data points and K classes
# should return three numpy arrays with shapes (D, K), (1, K), and (200,)
def discrimination_by_regression(X_train, Y_train,
                                 W_initial, w0_initial):
    eta = 1.0 / X_train.shape[0]
    iteration_count = 200

    W = W_initial
    w0 = w0_initial
    
    
    
    # your implementation starts below


    iteration=1
    objective_values=[]
    
    while True:
        
        y_predicted=sigmoid(X_train, W,w0)
        
        
        objective_values = np.append(objective_values,
                                 np.sum((Y_train-y_predicted)**2)*0.5) #fonksiyon düzelt
       
        
        W = W - eta * gradient_W(X, Y_train, y_predicted)
        
        
        w0 = w0 - eta * gradient_w0(Y_train, y_predicted)
       
        
        if iteration==iteration_count:
            break

        iteration = iteration + 1
        
    
    
    # your implementation ends above
    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)
print(W)
print(w0)
print(objective_values[0:10])


# In[10]:


fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("hw03_iterations.pdf", bbox_inches = "tight")


# In[11]:


# STEP 8
# assuming that there are N data points
# should return a numpy array with shape (N,)
def calculate_predicted_class_labels(X, W, w0):
    # your implementation starts below
    
    A=sigmoid(X,W,w0)
    
    y_predicted=  np.array([(np.argmax(A[i])+1) for i in range(len(X[:,0]))])
    
    # your implementation ends above
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test)


# In[12]:


# STEP 9
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, y_predicted):
    # your implementation starts below
    
    confusion_matrix = pd.crosstab(y_predicted.T, y_truth.T).values
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)


# In[ ]:





# In[ ]:




