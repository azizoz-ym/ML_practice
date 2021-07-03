#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


titanic_data = pd.read_excel(r"C:\Users\User\Desktop\Data Science\WHR\titanic.xlsx")
titanic_data


# In[10]:


titanic_data.head(3)


# In[11]:


titanic_data.tail(2)


# In[13]:


titanic_data.dtypes


# In[18]:


titanic_data.to_csv('titanic_csv_file.csv', encoding='utf-8')


# In[21]:


pd.read_csv("titanic_csv_file.csv")


# In[22]:


titanic_data.info()


# In[ ]:




