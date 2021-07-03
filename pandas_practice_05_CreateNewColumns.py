#!/usr/bin/env python
# coding: utf-8

# In[35]:


# tutorial source: 
# https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/05_add_columns.html

import pandas as pd

air_quality_file_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv"
air_quality = pd.read_csv(air_quality_file_path, index_col = 0, parse_dates = True)
air_quality


# In[9]:


air_quality["london_mg_per_cubic"] = air_quality["station_london"] * 1.882

air_quality


# In[13]:


air_quality["ratio_paris_antwerp"] = air_quality["station_paris"] / air_quality["station_antwerp"]
air_quality.head()


# In[25]:


air_quality_renamed = air_quality.rename(
    
    columns = {
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster"        
        
    }

)

# The rename() function can be used for both row labels and column labels. 
# Provide a dictionary with the keys the current names and the values the 
# new names to update the corresponding names.


# In[26]:


air_quality_renamed.head()


# In[27]:


air_quality.head()


# In[31]:


air_quality_renamed.rename(columns=str.lower)
air_quality_renamed.head()

# Details about column or row label renaming is provided in the 
# user guide section on renaming labels.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-rename


# In[36]:



#REMEMBER
#Create a new column by assigning the output to the DataFrame with a 
#new column name in between the [].

#Operations are element-wise, no need to loop over rows.

#Use rename with a dictionary or function to rename row labels or column names.


# In[ ]:




