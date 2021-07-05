#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


air_quality_no2_long_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv"
air_quality_no2 = pd.read_csv(air_quality_no2_long_path, parse_dates = True)

air_quality_no2 = air_quality_no2[["date.utc","location","parameter", "value" ]]
air_quality_no2


# In[9]:


air_quality_pm25_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_pm25_long.csv"
air_quality_pm25 = pd.read_csv(air_quality_pm25_path, parse_dates = True)
air_quality_pm25 = air_quality_pm25[["date.utc", "location", "parameter", "value"]]
air_quality_pm25


# In[12]:


air_quality = pd.concat([air_quality_no2, air_quality_pm25], axis = 0)
air_quality


# In[24]:


air_quality = air_quality.sort_values("date.utc")
air_quality


# In[27]:


air_quality_ = pd.concat([air_quality_pm25, air_quality_no2], keys = ["PM25", "NO2"])
air_quality_

#The existence of multiple row/column indices at the same time has not been 
#mentioned within these tutorials. Hierarchical indexing or MultiIndex is an advanced
#and powerful pandas feature to analyze higher dimensional data.

#Multi-indexing is out of scope for this pandas introduction. For the moment, 
#remember that the function reset_index can be used to convert any level of an index 
#to a column, e.g. air_quality.reset_index(level=0)

#To user guide
#Feel free to dive into the world of multi-indexing at the user guide section on advanced indexing.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced


# In[28]:


#More options on table concatenation (row and column wise) and how 
#concat can be used to define the logic (union or intersection) of the 
#indexes on the other axes is provided at the section on object concatenation.

# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#merging-concat


# In[31]:


air_quality_stations_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_stations.csv"
stations_coord = pd.read_csv(air_quality_stations_path)
stations_coord


# In[32]:


air_quality


# In[35]:


air_quality = pd.merge(air_quality, stations_coord, how ="left", on="location")
air_quality

#Using the merge() function, for each of the rows in the air_quality table, 
#the corresponding coordinates are added from the air_quality_stations_coord table.
#Both tables have the column location in common which is used as a key to combine the 
#information. By choosing the left join, only the locations available in the air_quality (left) 
#table, i.e. FR04014, BETR801 and London Westminster, end up in the resulting table. The merge 
#function supports multiple join options similar to database-style operations.


# In[36]:


air_quality_parameters_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_parameters.csv"
air_quality_parameters = pd.read_csv(air_quality_parameters_path)
air_quality_parameters


# In[43]:


air_quality = pd.merge(air_quality, air_quality_parameters,
                      how = "left", left_on="parameter", right_on="id")
air_quality.head()


# In[44]:


#Compared to the previous example, there is no common column name. 
#However, the parameter column in the air_quality table and the id column 
#in the air_quality_parameters_name both provide the measured variable in a common format. 
#The left_on and right_on arguments are used here (instead of just on) to make the link 
#between the two tables.


# In[45]:


#pandas supports also inner, outer, and right joins. 
#More information on join/merge of tables is provided in 
#the user guide section on database style merging of tables. 
#Or have a look at the comparison with SQL page.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#merging-join


# In[46]:


#REMEMBER
#Multiple tables can be concatenated both column-wise and row-wise using the concat function.

#For database-like merging/joining of tables, use the merge function.


# In[ ]:




