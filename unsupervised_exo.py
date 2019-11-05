
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

#load data

linesnames=np.genfromtxt('states.txt',delimiter='\n',dtype='U')
columnsnames=np.genfromtxt('crime_var.txt',delimiter='\n',dtype='U')
matrixX=np.loadtxt('USarrestdata.txt')


# ## US arrest data set
# 
# The data set contains the number of arrests per 100 000 residents for each of the following crimes: assault, murder and rape. It also contains the percentage of the population in each state living in urban area. 
# 
# 
# A. Perform a PCA on this data

# In[ ]:




# B. Do a biplot
# 
# 
# (a) Make a 2d plot with the names of the states on the pca1-pca2 plane. Use the function ax.annonate('label',(x,y),ha='center'). 'label' stands for the state name, (x,y) of the (pca1,pca2) values for the corresponding state. Use a "for" loop to plot all states. 
# [to put a "y" axis on the left and on the right, look at python_very_basic.ipynb]
# 
# (b) Represent the loading vectors on the plot. 

# In[14]:




# C. Compute the variance of the murder, assault, rape and urbanpop variables. 
# 
# Compute the variance of pca1 and pca2. 
# 
# What can you conclude? 
#    

# In[ ]:




# D. Do a 'Scree plot' 
# 
# Compute the percentage of the variance explained by the various pca components 
# Plot the result in a 'scree plot'

# In[ ]:




# E. Extra questions
# 
# 
# - Which variables influence most the first principal component? 
# 
# - Which variable influence most the second principal compontent? 
# 
# - Explain with your words how to interpret the loading vectors
# 
# - Is the assault variable more corrolated with rape or urban population? 
# 
# - Suppose that you do not like crimes in general. Where should you live? 
# 
# 
# 
# 

# ## Gradient descent - basics
# 
# (https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f)

# Generate data from a known model: 

# In[109]:

def myfunc(X,theta):
    x = np.array([1.0,X])
    return x.dot(theta)

x = np.linspace(0,10)
y = np.array([ myfunc(x[kk],np.array([3.0,1.0])) + np.random.normal(0,.4) for kk in range(len(x))])
plt.plot(x,y)
plt.show()


# Define a cost function

# In[110]:

def  cal_cost(theta,X,y,myfunc):    

    return cost

cal_cost(np.array([1.1,2.3]),x,y,myfunc)



# In[ ]:

Perform a gradient descent:


# In[114]:

def gradient_descent(X,y,theta,learning_rate=0.01,iterations=10000):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        
    return theta , cost_history, theta_history

gradient_descent(x,y,[1.2,3.4])


# ## Kullback Leibler divergence, perplexity and Shannon entropy 

# As a (useless) toymodel to understand t-SNE, pick-up random positions in 1d:

# In[8]:




# Choose one position (the "m" position), and build up Pm|l : (see slide 27)

# In[14]:




# In[15]:




# Check that the distribution is normalized:

# In[ ]:




# Compute the entropy of the distribution and the perplexity: 

# In[ ]:




# In[ ]:




# Play around with the sigma_l you choose (cf slide 27), and the spreadness of your points:

# Draw another set of points and compute the Kullback-Leibler divergence between the two associated distributions.
# When is the KL increasing or decreasing? 

# ## t-SNE : use a black box tSNE algorithm on the US arrest dataset

# In[ ]:



