#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


titanic_data = pd.read_excel(r"C:\Users\User\Desktop\Data Science\WHR\titanic.xlsx")
titanic_data.head()


# In[9]:


air_quality_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_long.csv"

air_quality_data = pd.read_csv(air_quality_path, index_col = "date.utc", parse_dates = True)
air_quality_data.head()


# In[10]:


titanic_data.sort_values( by = "Age" ).head()


# In[15]:


titanic_data.sort_values(by=["Pclass", "Age"],ascending = False)


# In[18]:


no2 = air_quality_data[air_quality_data["parameter"] == "no2"]
no2


# In[23]:


no2_subset = no2.sort_index().groupby(["location"]).head(2)
no2_subset


# In[27]:


no2_subset.pivot(columns = "location", values = "value")
#The pivot() function is purely reshaping of the data: 
#    a single value for each index/column combination is required.


# In[28]:


no2.head()


# In[32]:


no2.pivot(columns = "location", values = "value").plot(figsize = (12,4), subplots = True)


# In[41]:


air_quality_data.pivot_table(values="value",index="location",
                             columns ="parameter", aggfunc = "mean")

#In the case of pivot(), the data is only rearranged. When multiple values 
#need to be aggregated (in this specific case, the values on different 
#                       time steps) pivot_table() can be used, providing an 
#aggregation function (e.g. mean) on how to combine these values.


# In[43]:


#Pivot table is a well known concept in spreadsheet software. 
#When interested in summary columns for each variable separately as well,
#put the margin parameter to True:

air_quality_data.pivot_table(
    values = "value", index = "location", columns = "parameter",
    aggfunc = "mean", margins = True
)


#For more information about pivot_table(), see the user guide section on pivot tables.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-pivot


# In[46]:


#In case you are wondering, pivot_table() is indeed directly linked to groupby(). 
#The same result can be derived by grouping on both parameter and location:

air_quality_data.groupby(["parameter", "location"]).mean()


# In[49]:


no2_pivoted = no2.pivot(columns = "location", values = "value").reset_index()
no2_pivoted.head()


# In[53]:


no_2 = no2_pivoted.melt(id_vars = "date.utc")
no_2

#The pandas.melt() method on a DataFrame converts the data table from wide format to long format. 
#The column headers become the variable names in a newly created column.


# In[56]:


no_2_ = no2_pivoted.melt(id_vars = ["date.utc", "BETR801"])
no_2_


# In[62]:


no_2 = no2_pivoted.melt(
    id_vars = "date.utc",
    value_vars = ["BETR801", "FR04014", "London Westminster"],
    value_name = "NO_2",
    var_name = "id_location"

)

no_2.head()

#The result in the same, but in more detail defined:

#value_vars defines explicitly which columns to melt together

#value_name provides a custom column name for the values column instead of the default
#column name value

#var_name provides a custom column name for the column collecting the column header names. 
#Otherwise it takes the index name or a default variable

#Hence, the arguments value_name and var_name are just user-defined 
#names for the two generated columns. 
#The columns to melt are defined by id_vars and value_vars.


# In[63]:


#Conversion from wide to long format with pandas.melt() 
#is explained in the user guide section on reshaping by melt.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-melt


# In[64]:


#REMEMBER
#Sorting by one or more columns is supported by sort_values

#The pivot function is purely restructuring of the data, pivot_table supports aggregations

#The reverse of pivot (long to wide format) is melt (wide to long format)

#A full overview is available in the user guide on the pages about reshaping and pivoting.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping


# In[65]:


# DONE JULY 5, 2021, Sierra Manas, 18:49


# In[ ]:




