#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

import matplotlib.pyplot as plt


# In[21]:


air_quality_file_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv"
air_quality = pd.read_csv(air_quality_file_path, index_col = 0, parse_dates = True )
air_quality.head()


# In[22]:


air_quality.plot()


# In[23]:


air_quality["station_london"].plot()


# In[30]:


air_quality.plot.scatter(x = "station_london", y = "station_paris", alpha = 0.5)


# In[36]:


[
    method_name
    for method_name in dir(air_quality.plot)
    if not method_name.startswith("_")
]


# In[37]:


air_quality.plot.box()


# In[38]:


air_quality.plot.area()


# In[50]:


air_quality.plot.bar()


# In[51]:


axs = air_quality.plot.area(figsize = (12, 4), subplots = True)


# In[59]:


fig, axs = plt.subplots(figsize=(12, 4)) # Create an empty matplotlib Figure and Axes

air_quality.plot.area(ax=axs) # Use pandas to put the area plot on the prepared Figure/Axes

axs.set_ylabel("NO$_2$ concentration") # Do any matplotlib customization you like
fig.savefig("no2_concentration.png") # Save the Figure/Axes using the existing matplotlib method.


# In[62]:


# REMEMBER
# The .plot.* methods are applicable on both Series and DataFrames
# By default, each of the columns is plotted as a different element (line, boxplot,…)
# Any plot created by pandas is a Matplotlib object
# A full overview of plotting in pandas is provided in the visualization pages.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization


# In[ ]:




