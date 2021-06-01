#!/usr/bin/env python
# coding: utf-8

# **Roll No** : BECOC316
# 
# **Name** : Kedar Damkondwar
# 
# **Problem statement** :
# We have given a collection of points
# p1=[0.1,0.6], p2=[0.15,0.71], p3=[0.08,0.9], p4=[0.16,0.85], p5=[0.2,0.3],p6=[0.25,0.5], p7=[0.24,0.1], p8[0.3,0.2]
# Perform the k-means clustering with initial centroids as m1= P1 Cluster#1, and m2 = P6 Cluster#2
# 
# Answer the following
# 
# 1) Which cluster does P6 belong to?
# 
# 2) What is the population of cluster around m2?
# 
# 3) What is updated value of m1 and m2?

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


# In[2]:


dataset = {'x':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]}


# In[3]:


dataset = pd.DataFrame(dataset,columns=['x','y'])
dataset


# In[4]:


x = dataset.iloc[:,[0,1]].values
x


# In[5]:


centroids = np.array([x[0] , x[5]])
centroids


# In[6]:


k = KMeans(n_clusters = 2 , random_state = 0)
y_kmeans = k.fit_predict(x)
print('Cluster labels : ',y_kmeans)


# In[7]:


#clusters with initial centroids

plt.scatter(x[:,0], x[:,1], s = 100, c = 'cyan' , label = 'point')
plt.scatter(centroids[:,0],centroids[:,1], s = 150, c = 'black', label = 'centroid' , marker = '*')
plt.legend()


# In[8]:


print('(1) Which cluster does P6 belong to? =>  ',end='')
if y_kmeans[5]==0:
    print('m1 =',k.cluster_centers_[0])
else:
    print('m2 =',k.cluster_centers_[1])

print('(2) What is the population of cluster around m2? => ',list(y_kmeans).count(1))
print('(3) What is updated value of m1 and m2?')
print('    m1 = ',k.cluster_centers_[0],'\n    m2 = ',k.cluster_centers_[1])


# In[9]:


#clusters with updated centroids
plt.scatter(x[:,0], x[:,1], s = 100, c = 'cyan' , label = 'point')
plt.scatter(k.cluster_centers_[:,0],k.cluster_centers_[:,1], s = 150, c = 'black', label = 'centroid' , marker = '*')
plt.legend()


# In[ ]:




