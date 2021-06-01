#!/usr/bin/env python
# coding: utf-8

# **Roll No** : BECOC316
# 
# **Name** : Kedar Damkondwar
# 
# **Problem statement** :
#     Assignment on Linear Regression:
#         The following table shows the results of a recently conducted study on the correlation of the number of hours spent
#         driving with the risk of developing acute backache. Find the equation of the best fit line for this data.

# In[1]:


import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("hours.csv")
df.head()


# In[2]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[3]:


regressor = LinearRegression()
regressor.fit(x,y)


# In[4]:


hours = int(input('Enter the no of hours :'))


# In[5]:


eq=regressor.coef_*hours+regressor.intercept_
print(eq)


# In[6]:


plt.plot(x,y,'o')
plt.plot(x,regressor.predict(x),'b')
plt.show()


# In[ ]:




