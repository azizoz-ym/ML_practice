#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


air_quality_path = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv"
air_quality = pd.read_csv(air_quality_path)
air_quality = air_quality.rename(columns = {"date.utc":"datetime"})
air_quality


# In[9]:


air_quality.city.unique()


# In[12]:


#How to handle time series data with ease?
#Using pandas datetime properties

#I want to work with the dates in the column datetime as datetime objects instead of plain text

air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])
air_quality["datetime"]

#Initially, the values in datetime are character strings and do not provide
#any datetime operations (e.g. extract the year, day of the week,…). By applying
#the to_datetime function, pandas interprets the strings and convert these to 
#datetime (i.e. datetime64[ns, UTC]) objects. In pandas we call these datetime objects 
#similar to datetime.datetime from the standard library as pandas.Timestamp.


# In[13]:


#As many data sets do contain datetime information in one of the columns, 
#pandas input function like pandas.read_csv() and pandas.read_json() can do
#the transformation to dates when reading the data using the parse_dates parameter
#with a list of the columns to read as Timestamp:

#pd.read_csv("../data/air_quality_no2_long.csv", parse_dates=["datetime"])


# In[14]:


air_quality["datetime"].min(), air_quality["datetime"].max()


# In[15]:


air_quality["datetime"].max() - air_quality["datetime"].min()


# In[19]:


#The result is a pandas.Timedelta object, similar to datetime.timedelta from 
#the standard Python library and defining a time duration.

#The various time concepts supported by pandas are explained in the user
#guide section on time related concepts.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-overview


# In[21]:


air_quality["month"] = air_quality["datetime"].dt.month
air_quality


# In[22]:


#By using Timestamp objects for dates, a lot of time-related properties are provided by pandas. 
#For example the month, but also year, weekofyear, quarter,… All of these properties are accessible
#by the dt accessor.

#An overview of the existing date properties is given in the time 
#and date components overview table. More details about the dt accessor to 
#return datetime like properties are explained in a dedicated section on the dt accessor.

# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-components
# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dt-accessors


# In[24]:


air_quality.groupby(
    [air_quality["datetime"].dt.weekday, "location"])["value"].mean()


# In[26]:


fig, axs = plt.subplots(figsize = (12,4))

air_quality.groupby(air_quality["datetime"].dt.hour)["value"].mean().plot()


# In[34]:


air_quality.groupby(air_quality["datetime"].dt.hour)["value"].mean().plot(kind="bar",rot=0)


# In[40]:


#Datetime as index
#In the tutorial on reshaping, pivot() was introduced to reshape the data 
#table with each of the measurements locations as a separate column:

no_2 = air_quality.pivot(index="datetime", columns="location", values="value")
no_2


# In[37]:


no2.plot(figsize = (12,4),subplots = True)


# In[38]:


#By pivoting the data, the datetime information became the index of the table. 
#In general, setting a column as an index can be achieved by the set_index function.


# In[41]:


#Working with a datetime index (i.e. DatetimeIndex) provides powerful functionalities. 
#For example, we do not need the dt accessor to get the time series properties, 
#but have these properties available on the index directly:

no_2.index.year, no_2.index.weekday


# In[46]:


#Some other advantages are the convenient subsetting of time period or the adapted 
#time scale on plots. Let’s apply this on our data.

#Create a plot of the NO2 values in the different stations from the 20th of May
#till the end of 21st of May

no_2["2019-05-20":"2019-05-21"].plot()

#More information on the DatetimeIndex and the slicing by using strings is provided
#in the section on time series indexing.

# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-datetimeindex


# In[49]:


#Resample a time series to another frequency
#Aggregate the current hourly time series values to the monthly maximum value in each of the stations.

monthly_max = no_2.resample("M").max()
monthly_max


# In[50]:


#A very powerful method on time series data with a datetime index, is the ability to resample()
#time series to another frequency (e.g., converting secondly data into 5-minutely data).

#The resample() method is similar to a groupby operation:

#it provides a time-based grouping, by using a string (e.g. M, 5H,…) that defines the target frequency

#it requires an aggregation function such as mean, max,…


# In[51]:


#An overview of the aliases used to define time series frequencies is given in the offset aliases
#overview table.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases


# In[52]:


#When defined, the frequency of the time series is provided by the freq attribute:

monthly_max.index.freq


# In[62]:


no_2.resample("D").mean().plot(style="-o", figsize = (10,3));


# In[63]:


#More details on the power of time series resampling is 
#provided in the user guide section on resampling.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-resampling


# In[64]:


#REMEMBER
#Valid date strings can be converted to datetime objects using to_datetime function or as 
#part of read functions.

#Datetime objects in pandas support calculations, logical operations and convenient date-related
#properties using the dt accessor.

#A DatetimeIndex contains these date-related properties and supports convenient slicing.

#Resample is a powerful method to change the frequency of a time series.


# In[ ]:




