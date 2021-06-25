#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime


# In[2]:


# Task 1
# Examine the data, parse the time fields wherever necessary.
# Take the sum of the energy usage (use[kW]) to get per day usage and merge it with weather data

energy_df = pd.read_csv('energy_data.csv')
weather_df = pd.read_csv('weather_data.csv')

energy_df.head()


# In[3]:


weather_df.head()


# In[4]:


# parse time of weather
weather_df['time'] = pd.to_datetime(weather_df['time'], unit="s")
weather_df.head()


# In[5]:


# split up energy date and time into two columns

weather_df['Date'] = pd.to_datetime(weather_df['time']).dt.date
weather_df['Time'] = pd.to_datetime(weather_df['time']).dt.time
weather_df.rename(columns= {'time': 'Date & Time'}, inplace=True)

energy_df['Date'] = pd.to_datetime(energy_df['Date & Time']).dt.date
energy_df['Time'] = pd.to_datetime(energy_df['Date & Time']).dt.time

weather_df.head()


# In[6]:


energy_df.head()


# In[7]:


# Take the sum of the energy usage (use[kW]) to get per day usage and merge it with weather data
usage_df = energy_df.groupby('Date')[['use [kW]']].sum()
usage_df


# In[8]:


# have one weather row per day
# average of the number values per day
# exclude "icon" and "summary"
daily_df = weather_df.groupby('Date')[['temperature', 'humidity', 'visibility', 'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']].mean()


# In[9]:


daily_df


# In[10]:


df = pd.merge(daily_df, usage_df, on="Date")
df


# In[11]:


# Task 2
# Split the data obtained from step 1, into training and testing sets. The aim is to predict the usage
# for each day in the month of December using the weather data, so split accordingly. The usage
# as per devices should be dropped, only the “use [kW]” column is to be used for prediction from the dataset

# training set (days before December)
# splitting dataframe by row index
december = datetime.date(2014, 12, 1)
train_x = df[df.index < december] # df.iloc[:334]
train_y = train_x['use [kW]']
train_x = train_x.drop(['use [kW]'], axis=1)

# testing set (all days in December)
test_x = df[df.index >= december] # df.iloc[334:]
test_y = test_x['use [kW]']
test_x = test_x.drop(['use [kW]'], axis=1)


# In[12]:


train_x


# In[13]:


train_y


# In[14]:


test_x.head()


# In[15]:


test_y.head()


# In[16]:


# Task 3
# Linear Regression - Predicting Energy Usage:

# Set up a simple linear regression model to train, and then predict energy usage for each day in
# the month of December using features from weather data (Note that you need to drop the “use
# [kW]” column in the test set first). How well/badly does the model work? (Evaluate the correctness of your predictions based on the original “use [kW]” column). 
# Calculate the Root Mean Squared error of your model
# Finally generate a csv dump of the predicted values.
# Format of csv: Two columns, first should be the date and second should be the predicted value.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = train_x
y = train_y
reg = LinearRegression().fit(X, y)
predicted = reg.predict(test_x)

mean_squared_error(test_y, predicted, squared=False)


# In[17]:


# Finally generate a csv dump of the predicted values.
# Format of csv: Two columns, first should be the date and second should be the predicted value.

dates_list = [datetime.datetime.strftime(date, "%Y/%m/%d") for date in test_x.index]

csv_dump = pd.DataFrame(data=predicted, index=dates_list, columns=["Predicted Value"])
csv_dump.index.name = "Date"
csv_dump.to_csv('cse351_hw2_Lee_Cynthia_111737790_linear_regression.csv')


# In[18]:


# How well/badly does the model work?

# The root mean squared error is 8.740566311138375. A great model would have the root mean squared error very close to 0.
# A lower RMSE means a better model.
# This model was not that good.


# In[19]:


# Task 4

# Logistic Regression - Temperature classification:

# Using only weather data we want to classify if the temperature is high or low. Let's assume 
# temperature greater than or equal to 60 is ‘high’ and below 60 is ‘low’. Set up a logistic
# regression model to classify the temperature for each day in the month of December. Calculate the F1 score for the model.
# Finally generate a csv dump of the classification (1 for high, 0 for low)
# Format: Two columns, first should be the date and second should be the classification (1/0).

# treshold changed to 35

temp_df = daily_df.copy()
temp_df.loc[temp_df['temperature'] >= 35, 'temp'] = 1 # high
temp_df.loc[temp_df['temperature'] < 35, 'temp'] = 0 # low
temp_df.drop(['temperature'], axis=1)


# In[20]:


# splitting the data

# training set (days before December)
# splitting dataframe by row index
december = datetime.date(2014, 12, 1)
train_x = temp_df[temp_df.index < december]
train_y = train_x['temp']
train_x = train_x.drop(['temp'], axis=1)

# testing set (all days in December)
test_x = temp_df[temp_df.index >= december]
test_y = test_x['temp']
test_x = test_x.drop(['temp'], axis=1)


# In[21]:


train_x


# In[22]:


# logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# scale data
scaler = StandardScaler().fit(train_x)
s_train_x = scaler.transform(train_x)
s_test_x = scaler.transform(test_x)

X = s_train_x
y = train_y
reg = LogisticRegression().fit(X, y)
predicted = reg.predict(s_test_x)

f1_score(test_y, predicted)


# In[23]:


# Finally generate a csv dump of the classification (1 for high, 0 for low)
# Format: Two columns, first should be the date and second should be the classification (1/0).

dates_list = [datetime.datetime.strftime(date, "%Y/%m/%d") for date in test_x.index]

csv_dump = pd.DataFrame(data=predicted, index=dates_list, columns=["Classification"])
csv_dump.index.name = "Date"
csv_dump.to_csv('cse351_hw2_Lee_Cynthia_111737790_logistic_regression.csv')


# In[24]:


# Task 5

# Energy usage data Analysis:

# We want to analyze how different devices are being used in different times of the day.
# - Is the washer being used only during the day?
# - During what time of the day is AC used most?
# There are a number of questions that can be asked.
# For simplicity, let’s divide a day in two parts:
# - Day : 6AM - 7PM
# - Night: 7PM - 6AM
# Analyze the usage of any two devices of your choice during the ‘day’ and ‘night’. Plot these
# trends. Explain your findings.

energy_df.head()


# In[25]:


# assign day into to parts
day_energy_df = energy_df.copy()

start = datetime.time(6,0)
end = datetime.time(19,0)
time = day_energy_df['Time']
day_energy_df.loc[((start <= time) | (time < end)), 'Day'] = 1 # day

start = datetime.time(19,0)
end = datetime.time(6,0)
day_energy_df.loc[((start <= time) | (time < end)), 'Day'] = 0 # night

day_energy_df = day_energy_df.drop(['Time', 'Date & Time'], axis=1)
day_energy_df


# In[26]:


# Analyze the usage of any two devices of your choice during the ‘day’ and ‘night’. 
# Plot these trends. Explain your findings.

# Choosen devices: Furnace, Washer

devices_df = day_energy_df.filter(['Washer [kW]','Furnace [kW]','Date','Day'], axis=1)
devices_df


# In[27]:


washer_df = day_energy_df.filter(['Washer [kW]','Date','Day'], axis=1)
furnace_df = day_energy_df.filter(['Furnace [kW]','Date','Day'], axis=1)


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plot = sns.countplot(x="Day", data=devices_df)
plot.set_title("Usage of Washer and Furnace During Time of Day")
plot.set_xticklabels(["Night", "Day"])
plot.set_xlabel("Time of Day")


# In[29]:


# Is the washer only being used during the day? No
# Used most during the day

plot = sns.catplot(data=washer_df, x="Day", y="Washer [kW]")
plot.set_titles("Usage of Washer and Furnace During Time of Day")
plot.set_xticklabels(["Night", "Day"])
plot.set_xlabels("Time of Day")


# In[30]:


# The furnace is used both day and night

plot = sns.catplot(data=furnace_df, x="Day", y="Furnace [kW]")
plot.set_titles("Usage of Washer and Furnace During Time of Day")
plot.set_xticklabels(["Night", "Day"])
plot.set_xlabels("Time of Day")


# In[31]:


plt.figure(figsize=(15, 7))
sns.lineplot(data=washer_df, x="Date", y="Washer [kW]").set_title("Usage of Washer")

# Washer has a lot of small periods of not being used 
# there is a pattern of not being used and then a spike and then not being used again
# perhaps this household does laundry every 1 or 2 weeks which explains this pattern, laundry is not done every day
# during the spikes, laundry is done throughout the year as shown with the washer usage


# In[32]:


plt.figure(figsize=(15, 7))
sns.lineplot(data=furnace_df, x="Date", y="Furnace [kW]").set_title("Usage of Furnace")

# Furnace hasn't had much usage at all during some time periods after 2014-5 and before 2014-7
# also during the time period around the end of 2014-9
# perhaps this household has taken a vacation during the summer months and left the house
# during the vacation they did not use their kitchen/furnace

# Furnace usage kW is higher during the months of 1-3 (January to March) compared to 7-11 (July to November)
# perhaps bigger and warmer meals are cooked furing the winter time compared to the spring and summer 
# bigger and warmer meals would require more furnace usage


# In[33]:


washer_day_df = washer_df.loc[washer_df['Day'] == 1]
washer_night_df = washer_df.loc[washer_df['Day'] == 0]

plt.figure(figsize=(15, 7))
sns.lineplot(data=washer_day_df, x="Date", y="Washer [kW]", color="tomato").set_title("Usage of Washer During the Day")


# In[34]:


plt.figure(figsize=(15, 7))
sns.lineplot(data=washer_night_df, x="Date", y="Washer [kW]", color="slateblue").set_title("Usage of Washer During the Night")

# washer is used less frequently during the night
# perhaps washing clothes is easier to deal during the day if some delicate clothes need to be hung up to dry
# or perhaps this household prefers to do laundry during the day


# In[35]:


furnace_day_df = furnace_df.loc[furnace_df['Day'] == 1]
furnace_night_df = furnace_df.loc[furnace_df['Day'] == 0]

plt.figure(figsize=(15, 7))
sns.lineplot(data=furnace_day_df, x="Date", y="Furnace [kW]", color="tomato").set_title("Usage of Furnace During the Day")


# In[36]:


plt.figure(figsize=(15, 7))
sns.lineplot(data=furnace_night_df, x="Date", y="Furnace [kW]", color="slateblue").set_title("Usage of Furnace During the Night")

# furnace usage used both during the day and night, probably because of cooking meals during the day and night
# ex. breakfast during the day, lunch, dinner during the night


# In[37]:


plt.figure(figsize=(7, 4))
sns.boxplot(x="Washer [kW]", data=washer_day_df).set_title("Usage of Washer During the Day")


# In[38]:


plt.figure(figsize=(7, 4))
sns.boxplot(x="Washer [kW]", data=washer_night_df).set_title("Usage of Washer During the Night")


# In[39]:


plt.figure(figsize=(7, 4))
sns.boxplot(x="Furnace [kW]", data=furnace_day_df).set_title("Usage of Furnace During the Day")


# In[40]:


plt.figure(figsize=(7, 4))
sns.boxplot(x="Furnace [kW]", data=furnace_night_df).set_title("Usage of Furnace During the Night")
# furnace usage during the night has an average of more datapoints with higher values than during the day
# perhaps dinner requires more furnace usage than breakfast 
# as dinner meals tend to be more hearty and bigger than breakfast meals


# In[ ]:




