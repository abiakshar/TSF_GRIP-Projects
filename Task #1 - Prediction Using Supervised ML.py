#!/usr/bin/env python
# coding: utf-8

# ### Abirami Baskaran
# # Task #1 Prediction using Supervised ML
# ## Predict the percentage of an student based on the no.of study hours

# In[9]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[13]:


# Create dataframe 
student_data=pd.DataFrame({'hours':  [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
'scores' : [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]})
student_data.head()


# In[14]:


student_data.shape


# In[48]:


# Create scatterplot for get on idea of distribution of data points
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Hours/day',fontsize=12)
plt.ylabel('Scores',fontsize=12)
plt.title('Percentage of student based on no. of study hours',fontsize=15)
plt.scatter(student_data.hours,student_data.scores,color='orange')


# In[34]:


# Create a Linear Regression object
reg=linear_model.LinearRegression()

# Train Linear Regression model 
reg.fit(student_data[['hours']],student_data.scores)


# In[52]:


# Predicting score if a student studies 9.25 hrs/day
reg.predict([[9.25]])


# In[36]:


# Finding slope of the equation
reg.coef_


# In[37]:


# Finding intercept of the equation
reg.intercept_


# In[60]:


# y=mx+c - equation of the line
9.77580339*9.25+2.483673405373196


# In[59]:


# Create linear regression line for the data points
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Hours/day',fontsize=12)
plt.ylabel('Scores',fontsize=12)
plt.title('Linear Regression',fontsize=15)
plt.scatter(student_data.hours,student_data.scores,color='orange')
plt.plot(student_data.hours,reg.predict(student_data[['hours']]),color='green')


# In[57]:


# Predict some of the scores using Linear Regression
reg.predict([[2.6],[3.9],[4.1],[5.2],[6.3],[7.5],[8.2],[9.4]])


# In[ ]:




