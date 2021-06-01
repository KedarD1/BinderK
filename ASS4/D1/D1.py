#!/usr/bin/env python
# coding: utf-8

# **Roll No** : BECOC316
# 
# **Name** : Kedar Damkondwar
# 
# **Problem statement** :
#     Assignment on Decision Tree Classifier:
#         A dataset collected in a cosmetics shop showing details of customers and whether or not they responded to a special offer to buy a new lip-stick is shown in table below. Use this dataset to build a decision tree, with Buys as the target variable, to help in buying lip-sticks in the future. Find the root node of decision tree. According to the decision tree you have made from previous training data set, what is the decision for the test data:
#         [Age < 21, Income = Low,Gender = Female, Marital Status = Married]?

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv("cosmetics.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,4].values


# In[3]:


from sklearn.preprocessing import LabelEncoder
lable_en=LabelEncoder()
x=x.apply(LabelEncoder().fit_transform)
print(x)


# In[4]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x,y)


# In[5]:


from sklearn.model_selection import cross_val_score
dt_scores=cross_val_score(dt,x,y,cv=3)
dt_scores


# In[6]:


dt_scores.mean()


# In[7]:


x_ex=np.array([1,1,0,0])
y_pred=dt.predict([x_ex])


# In[8]:


y_pred


# In[9]:


dt_en=DecisionTreeClassifier(criterion="entropy")
dt_en.fit(x,y)


# In[10]:


dt_scores_en=cross_val_score(dt,x,y,cv=3)
dt_scores_en


# In[11]:


dt_scores_en.mean()


# In[12]:


y_pred_en=dt_en.predict([x_ex])
y_pred_en


# In[ ]:




