#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

titanic_data = pd.read_excel(r"C:\Users\User\Desktop\Data Science\WHR\titanic.xlsx")
titanic_data.head()


# In[10]:


titanic_data["Age"].mean()


# In[11]:


titanic_data[["Age", "Fare"]].median()


# In[ ]:





# In[12]:


titanic_data[["Age", "Fare"]].describe()


# 

# In[13]:


titanic_data.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        "Fare" : ["min", "max", "median", "mean"],
    }



)


# In[14]:


#What is the average age for male versus female Titanic passengers?
titanic_data[["Age", "Sex"]].groupby("Sex").mean()


# In[16]:


titanic_data.groupby("Sex").mean()


# In[17]:


titanic_data.groupby("Sex")["Age"].mean()


# In[22]:


titanic_data.groupby(["Sex", "Pclass"])["Fare"].count()


# In[20]:


titanic_data.groupby(["Sex", "Pclass"])["Fare"].mean()


# In[28]:


titanic_data["Pclass"].value_counts()
#The value_counts() method counts the number of records for each category in a column.


# In[24]:


titanic_data["Pclass"].count()


# In[26]:


titanic_data.groupby("Sex")["Name"].count()


# In[27]:


titanic_data.groupby("Sex").count()


# In[30]:


titanic_data.groupby("Pclass")["Pclass"].count()


# In[ ]:


#REMEMBER
#Aggregation statistics can be calculated on entire columns or rows

#groupby provides the power of the split-apply-combine pattern

#value_counts is a convenient shortcut to count the number of entries in each category of a variable

