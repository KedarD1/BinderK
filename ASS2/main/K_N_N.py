#!/usr/bin/env python
# coding: utf-8

# **Roll No** : BECOC316
# 
# **Name** : Kedar Damkondwar
# 
# **Problem statement** :
#     Assignment on k-NN Classification:
#         In the following diagram let blue circles indicate positive examples and orange squares indicate negative examples. 
#         We want to use k-NN algorithm for classifying the points. If k=3, find the class of the point (6,6). 
#         Extend the same example for Distance-Weighted k-NN and Locally weighted Averaging

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


X=[[2,4],[4,6],[4,4],[4,2],[6,4],[6,2]]


# In[3]:


y=['Negative','Negative','Positive','Negative','Negative','Positive']


# In[4]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[5]:


knn.fit(X,y)


# In[6]:


new_data=[[6,6]]


# In[7]:


print("KNN for [6,6]",knn.predict(new_data))


# In[8]:


ypred_1=knn.predict(X)
print("KNN for all data",ypred_1)


# In[9]:


print("Actual",y)


# In[10]:


print("Accuracy of KNN",str(accuracy_score(y,ypred_1)))


# In[11]:


print("Confusion matrix of KNN",confusion_matrix(y,ypred_1))


# In[12]:


from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()
clf.fit(X,y)
clf.centroids_
ypred_3=clf.predict(X)
print(ypred_3)
print(y)


# In[13]:


print(accuracy_score(y,ypred_3))


# In[14]:


print(confusion_matrix(y,ypred_3))


# In[15]:


print(clf.predict(new_data))


# In[ ]:




