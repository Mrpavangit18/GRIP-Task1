#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


link= "http://bit.ly/w-data"
data=pd.read_csv(link)


# In[3]:


print("Data imported successfully")
data.head(10)


# In[5]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


x=data.iloc[:,:-1].values  
y=data.iloc[:,1].values  


# In[7]:


x


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[9]:


x_test


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


LR=LinearRegression()
LR.fit(x_train,y_train)
print("We have successfully train the algo")


# In[12]:


# Plotting the regression line
line = LR.coef_*x+LR.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# In[13]:


# Predicting the scores
y_pred = LR.predict(x_test) 


# In[14]:


y_pred


# In[15]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




