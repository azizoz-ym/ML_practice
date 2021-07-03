#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd


# In[14]:


df = pd.DataFrame(
    {
        "Name": ["Askar", "Kurmanbek", "Almazbek"],
        "Age": [76, 71, 64],
        "Sex": ['male', 'male', 'male']
        
    }
)

df


# In[15]:


df["Age"]


# In[21]:


ages = pd.Series([76, 71, 64], name = "Ages")
ages


# In[28]:


max_age = df["Age"].max()
max_age


# In[29]:


ages.max()


# In[31]:


df.describe()


# In[ ]:




