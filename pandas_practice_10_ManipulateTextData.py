#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd


# In[37]:


titanic_data_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
titanic_data = pd.read_csv(titanic_data_path)
titanic_data


# In[38]:


#How to manipulate textual data?
#Make all name characters lowercase.

titanic_data["Name"].str.lower()

#To make each of the strings in the Name column lowercase, select the Name column 
#(see the tutorial on selection of data), add the str accessor and apply the lower method.
#As such, each of the strings is converted element-wise. 


# In[39]:


#Create a new column Surname that contains the surname of the passengers by extracting the 
#part before the comma.

titanic_data["Name"].str.split(",")


# In[40]:


titanic_data["Surname"] = titanic_data["Name"].str.split(",").str.get(0)
titanic_data["Surname"]


# In[41]:


#More information on extracting parts of strings is available in the user guide 
#section on splitting and replacing strings.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#text-split


# In[42]:


#Extract the passenger data about the countesses on board of the Titanic.
titanic_data["Name"].str.contains("Countess")


# In[65]:


titanic_data[titanic_data["Name"].str.contains("Countess")]

#(Interested in her story? See Wikipedia!)
# https://en.wikipedia.org/wiki/No%C3%ABl_Leslie,_Countess_of_Rothes

#More powerful extractions on strings are supported, as the Series.str.contains() 
#and Series.str.extract() methods accept regular expressions, but out of scope of this tutorial.


# In[66]:


titanic_data["Name"].str.len()


# In[67]:


titanic_data["Name"].str.len().idxmax()


# In[73]:


titanic_data.loc[titanic_data["Name"].str.len().idxmax(), ["Name","Age", "Survived", "Pclass"]]


# In[74]:


titanic_data["Sex_short"] = titanic_data["Sex"].replace({"male":"M", "female":"F"})
titanic_data["Sex_short"]


# In[75]:


titanic_data


# In[76]:


#Whereas replace() is not a string method, it provides a convenient way 
#to use mappings or vocabularies to translate certain values. It requires a dictionary 
#to define the mapping {from : to}.

#There is also a replace() method available to replace a specific set of characters. 
#However, when having a mapping of multiple values, this would become:

# titanic["Sex_short"] = titanic["Sex"].str.replace("female", "F")
# titanic["Sex_short"] = titanic["Sex_short"].str.replace("male", "M")


# In[77]:


#REMEMBER
#String methods are available using the str accessor.

#String methods work element-wise and can be used for conditional indexing.

#The replace method is a convenient method to convert values according to a given dictionary.

#To user guide
#full overview is provided in the user guide pages on working with text data.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#text


# In[ ]:




