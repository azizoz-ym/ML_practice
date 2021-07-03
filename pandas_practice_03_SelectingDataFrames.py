#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
titanic_file_path = r'C:\Users\User\Desktop\Data Science\WHR\titanic.xlsx'
titanic_data = pd.read_excel(titanic_file_path)
titanic_data.head()


# In[75]:


ages = titanic_data["Age"]
ages.head()


# In[76]:


type(titanic_data["Age"])


# In[77]:


titanic_data["Age"].shape


# In[78]:


titanic_data[["Name","Age", "Sex", "Survived"]]


# In[79]:


type(titanic_data[["Name", "Age", "Sex"]])


# In[80]:


titanic_data[["Name", "Age", "Sex"]].shape


# In[81]:


below_18 = titanic_data[titanic_data["Age"] < 18]
below_18.head()


# In[82]:


below_18.shape


# In[102]:


yo_18 = titanic_data[titanic_data["Age"] == 18]
yo_18.shape


# In[105]:


class_2_3 = titanic_data[titanic_data["Pclass"].isin([2,3])]
class_2_3.shape


# In[111]:


class_23 = titanic_data[(titanic_data["Pclass"] == 2) | (titanic_data["Pclass"] == 3)]
class_23.shape


# In[115]:


age_known = titanic_data[titanic_data["Age"].notna()]
age_known.shape


# In[130]:


children_names = titanic_data.loc[titanic_data["Age"] < 18, ["Name", "Sex", "Age", "Survived" ]]
children_names


# In[131]:


titanic_data.iloc[9:25, 2:5]


# In[ ]:




