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


# In[7]:


sc=cross_val_score(regressor,x,y,cv=3,scoring='r2')
sc.mean()


# In[8]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge,LassoCV,Lasso,ElasticNet,LinearRegression

X=load_diabetes().data
y=load_diabetes().target


# In[9]:


lr=LinearRegression(normalize=True)
lr_score=cross_val_score(lr,X,y,cv=10)
lr_score.mean()


# In[10]:


al=[0.5,0.1,1,2,5,10,0.05,0.005]
for i in range (len(al)):
    ridge=Ridge(alpha=al[i])
    lr_score=cross_val_score(ridge,X,y,cv=10)
    print(al[i],lr_score.mean())


# In[11]:


ridge= Ridge(alpha=0.05)
ridge_score=cross_val_score(ridge,X,y,cv=10,)
ridge_score.mean()


# In[12]:


lassocv=LassoCV(alphas=(1,0.1,0.5,0.05,0.0025,0.0001),normalize=True)
lassocv.fit(X,y)


# In[13]:


lasso= Lasso(alpha=0.005)
lasso_score=cross_val_score(lasso,X,y,cv=10)
lasso_score.mean()


# In[14]:


en=ElasticNet(alpha=0.001,l1_ratio=0.8,normalize=True)
en_score=cross_val_score(en,X,y,cv=10)
en_score.mean()


# In[15]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
reg = ['linear', 'lasso', 'ridge', 'elastic']
score = [lr_score.mean(),lasso_score.mean(),ridge_score.mean(),en_score.mean()]
ax.bar(reg,score)
plt.ylim(0.462,0.465)
plt.show()


# In[ ]:




