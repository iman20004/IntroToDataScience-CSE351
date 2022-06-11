#!/usr/bin/env python
# coding: utf-8

# # CSE351 HW2: Prediction/Modelling
# ## Predict electricity usage based on weather conditions
# **Iman Ali (112204305)**

# In[8]:


# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# ## Load the data

# In[9]:


# Use pandas to load data frames
energy_data = pd.read_csv('energy_data.csv')
weather_data = pd.read_csv('weather_data.csv')


# ## Task 1

# ### (Examine data, parse time fields, sum energy usage per day & merge with weather data)

# In[10]:


############################################ Examine the data #####################################################

# Display first 5 rows of the energy dataset
energy_data.head()


# In[11]:


# Display first 5 rows of the weather dataset
weather_data.head()


# In[12]:


# A concise summary of the Energy DataFrame. 
# Check which properties are being represented and what their types are.
energy_data.info()


# In[13]:


# A concise summary of the Weather DataFrame. 
# Check which properties are being represented and what their types are.
weather_data.info()


# In[14]:


############################################ Parse time fields ####################################################

########## Energy Data ###########
# Extract time from 'Date & Time' field (Add it as a new field)
energy_data['time'] = pd.to_datetime(energy_data['Date & Time']).dt.time
energy_data.head()


# In[15]:


########## Weather Data ###########

# First convert 'time' to 'Date & Time' format so it is compatible with energy_data (adding as new field)
weather_data['Date & Time'] = pd.to_datetime(weather_data['time'], unit="s")

# Then extract time from it (same as energy_data time format HH:mm::ss)
weather_data['time'] = weather_data['Date & Time'].dt.time
weather_data.head()


# In[16]:


############################## Sum energy usage (Use [kW]) to get per day usage ###################################

# To get per day usage, we first need to group the energy_data rows by date.
# To do that, we need to extract date from the 'Date & Time' field and add it as another column.
energy_data['Date'] = pd.to_datetime(energy_data['Date & Time']).dt.normalize()
energy_data.head()


# In[17]:


# Now group by date and sum the energy usage
daily_usage = energy_data.groupby("Date")
daily_usage = daily_usage[['use [kW]']].sum()
daily_usage


# In[18]:


##################################### Merge daily usage with weather data #########################################

# We have already calculated the daily usage, but now we need to get daily weather data.
# Therefore group weather data by date and take averge of values.

# To do that, extract date from 'Date & Time' field (adding it as new column)
weather_data['Date'] = pd.to_datetime(weather_data['Date & Time']).dt.normalize()
weather_data.head()


# In[19]:


# Group weather data by date and mean over all numeric values
daily_weather = weather_data.groupby("Date").mean()
daily_weather


# In[20]:


# Merge daily_usage with daily_weather
daily_data = pd.merge(daily_usage, daily_weather, on="Date")
daily_data


# ## Task 2
# ### (Split into training and testing sets)

# In[21]:


# Training set => day in January - November
# Testing set => days in December

# Extract training set
training = daily_data[daily_data.index < '2014-12-01']

# Target variable 'use [kW]' as y
# Therefore Y_train has values of 'use [kw]', and X_train with all other fields
Y_train = training[['use [kW]']]
X_train = training.loc[:, training.columns != 'use [kW]']


# Display X_train
X_train


# In[22]:


# Display Y_train (answers)
Y_train


# In[23]:


# Extract testing set (december)
testing = daily_data[daily_data.index >= '2014-12-01']

# Similar to training data,
# Y_test with values of 'use [kw]'', and X_test with all other fields
Y_test = testing[['use [kW]']]
X_test = testing.loc[:, training.columns != 'use [kW]']


# Display X_test
X_test


# In[25]:


# Display Y_test (answers)
Y_test


# ## Task 3 
# ## (Linear Regression - Predicting Energy Usage)

# In[27]:


################################# Set up a simple linear regression model to train ################################

# Use linear regression from the sklearn library 
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Use the coefficient and intercept to determine the y = mx + c linear fit
print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)


# In[41]:


############################ Predict energy usage for each day in the month of December ############################

# Use the model to predict values of electricty usage for the testing set
usage_pred = lin_reg.predict(X_test)
usage_pred = usage_pred.flatten()
usage_pred

# Check csv dump for predicted values as well.


# In[42]:


# Actual values vs Predicted
x = pd.DataFrame({'Actual': Y_test['use [kW]'].tolist(),'Predicted': usage_pred })
x


# In[43]:


#################################### How well/badly does the model work ##########################################

# Evaluate the correctness of your predictions based on the original “use [kW]” column
lin_reg.score(X_test, Y_test)


# In[44]:


########################################### Root mean squared error #############################################
mean_squared_error(Y_test, usage_pred, squared=False)


# In[65]:


# According to the sklearn library:
# Root mean squared error return: the best value is 0.0 

# Lower values of RMSE indicate better fit.
# Since we have quite a high value of 8.74 (not close to zero) that means our model was not the best

# But given that we dont want to overfit or underfit the data by overlearning or underlearning respectively,
# This model does a decent job at predicting the values


# In[46]:


##################################### A csv dump of the predicted values ########################################

# Format of csv: Two columns, first date and second predicted value
data = pd.DataFrame({'Date':X_test.index, 'Predicted Usage':usage_pred})
data.to_csv('cse351_hw2_Ali_Iman_112204305_linear_regression.csv')


# ## Task 4
# ## Logistic Regression - Temperature classification

# In[54]:


# Training set => day in January - November
# Testing set => days in December

# Extract training set (not december)
training = daily_data[daily_data.index < '2014-12-01']

# temperature class that contains 1 for high temp (>=35) and 0 for low temp (< 35)
temperature_class = [1 if temp >= 35 else 0 for temp in training.temperature]

# Target variable 'temperature_class' as y
# This field is not in the data frame, therefore training set will act as the X_train
# And temperature_class array will act as the Y_train
Y_train = temperature_class
X_train = training

# Display X_train
X_train


# In[55]:


# Extract testing set (december)
testing = daily_data[daily_data.index >= '2014-12-01']

# Similar to training data, testing data will act as X_test
X_test = testing

# Y_test will be temperature_class of testing set
Y_test = [1 if temp >= 35 else 0 for temp in testing.temperature]


# In[56]:


################################### Set up a logistic regression model ###########################################

# Use logistic regression from the sklearn library 
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, Y_train)


# In[60]:


############################### Classify temperature class for each day in December ##############################

# Use the model to predict the temperature class (high/low) for the testing set
temp_class_pred = log_reg.predict(X_test)
temp_class_pred


# In[61]:


# Actual values vs Predicted
x = pd.DataFrame({'Actual': Y_test,'Predicted': temp_class_pred })
x


# In[62]:


############################################# F1 score for the model ##############################################

f1_score(Y_test, temp_class_pred)


# In[63]:


# Our logistic regression model did a pretty decent job with a high f1 score of 0.82
# It is pretty close to a perfect score of 1.0


# In[64]:


##################################### A csv dump of the predicted values ########################################

# Format of csv: Two columns, first date and second classification
data = pd.DataFrame({'Date':X_test.index, 'Classification':temp_class_pred})
data.to_csv('cse351_hw2_Ali_Iman_112204305_logistic_regression.csv')


# ## Task 5
# ## Energy Usage Data Analysis
# ####  Analyze how different devices are being used in different times of the day

# In[422]:


######################################### Divide day in two parts ###############################################

# Day: 6AM - 7PM
# Night: 7PM - 6AM

# Add new field into energy_data frame; 
# 'Time_of_day' which is 1 for day values (6AM - 7PM) and 0 for night values (7PM - 6AM)
start_time = pd.to_datetime("06:00:00").time()
end_time = pd.to_datetime("19:00:00").time()

energy_data['Time_of_day'] = [1 if (time >= start_time and time < end_time) else 0 for time in energy_data.time]
energy_data


# In[423]:


# Group energy_data by 'Date' and then 'Day'
# Therefore, we get day and night usage of each appliance, every day.
daily_energy = energy_data.groupby(['Date', 'Time_of_day']).sum()
daily_energy


# In[424]:


########################################### Usage of any two devices ##############################################

# We will analyze the usage of 'Microwave' and 'Fridge' during the ‘day’ and ‘night’

usage_data = daily_energy[['Microwave (R) [kW]', 'Fridge (R) [kW]']]
usage_data


# In[425]:


# Get total usage (sum) of microwave and fridge for day and night

day_usage_sum = usage_data.groupby('Time_of_day').sum()
day_usage_sum['Total Usage [kW]'] = day_usage_sum['Microwave (R) [kW]'] + day_usage_sum['Fridge (R) [kW]']
day_usage_sum


# In[426]:


# Visualization: plot Total Usage (sum of both applainces) during different times of the day

graph = day_usage_sum.plot.bar(y='Total Usage [kW]', figsize=(10,8), legend=False, color=['#00008B','#FFD500'])
plt.title("Total Usage during different times of day",fontweight='bold', fontsize=20)
graph.set_xticklabels(["Night", "Day"])
graph.set_xlabel("Time of Day", fontsize=15)
graph.set_ylabel("Total Usage [kW]", fontsize=15)


# In[427]:


# Findings:

# Total usage is greater during the day as compared to the night.
# Appliances are being used more frequently from 6AM to 7PM

# Explanation: This is a very expected result because people use more appliances when they are awake in the day
# as compared to asleep during the night.


# In[428]:


# Visualization: plot total usage of each appliance day and night

graph = day_usage_sum.plot.bar(y=day_usage_sum.index, figsize=(10,8))
plt.title("Total Usage of Microwave & Fridge Day vs Night",fontweight='bold', fontsize=15)
graph.set_xticklabels(["Night", "Day"])
graph.set_xlabel("Time of Day", fontsize=15)
graph.set_ylabel("Total Usage [kW]", fontsize=15)
graph.legend(loc=2, prop={'size': 15})


# In[429]:


# Visualization: Same as graph above, but different visualization
# Usage of both appliances stacked on top of each other, showing total usage also.

graph = day_usage_sum.plot.bar(y=day_usage_sum.index, stacked=True, figsize=(10,8))
plt.title("Total Usage of Microwave & Fridge Day vs Night",fontweight='bold', fontsize=15)
graph.set_xticklabels(["Night", "Day"])
graph.set_xlabel("Time of Day", fontsize=15)
graph.set_ylabel("Total Usage [kW]", fontsize=15)
graph.legend(loc=2, prop={'size': 15})


# In[430]:


# Findings:

# The above 2 bargraphs show that both the Fridge and Microwave are being used during the day as well as the night.

# However, for both, the usage is greater during the day as compared to night time.

# We can also conclude that the usage of fridge is much more compared to the microwave, irrespective of 
# time of day.

# Explanation: Again, more people are awake during the day and use more devices, such as the microwave to heat 
# their food.
# The fact that fridges are always using electricty to maintain a low temperature inside must be a major factor  
# contributing to usage during the night and also having more usage than the microwave.


# In[431]:


# Visualization: Compare Usage during different times of the day

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))


# Microwave
ax1 = day_usage_sum.plot.bar(y='Microwave (R) [kW]', ax=ax1, legend=False, 
                             color=['#00008B','#FFD500'], title="Microwave Usage during day & night")
ax1.set_xticklabels(["Night", "Day"])
ax1.set_xlabel("Time of Day", fontsize=15)
ax1.set_ylabel("Total Usage [kW]", fontsize=15)


# Fridge
ax2 = day_usage_sum.plot.bar(y='Fridge (R) [kW]', ax=ax2, legend=False, 
                             color=['#00008B','#FFD500'], title="Fridge Usage during day & night")
ax2.set_xticklabels(["Night", "Day"])
ax2.set_xlabel("Time of Day", fontsize=15)
ax2.set_ylabel("Total Usage [kW]", fontsize=15)

ax1.title.set_size(18)
ax1.title.set_weight('bold')
ax2.title.set_size(18)
ax2.title.set_weight('bold')


# In[432]:


# Findings:

# Microwave has skewed distribution of usage. 
# Much more usage during day compared to night

# Explanation:
# Awake people warming up food compared to sleeping individuals

# Fridge has much more even distrbution of usage.

# Explanation:
# Almost same amount of electricity required to keep the temperature low and maintain it.


# In[433]:


# Visualization: Scatter plot to check spread of data

scatter = usage_data.reset_index()
day_scatter = scatter[scatter['Time_of_day'] == 1]
day_scatter


# In[434]:


night_scatter = scatter[scatter['Time_of_day'] == 0]
night_scatter


# In[435]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=day_scatter['Date'], y=day_scatter['Microwave (R) [kW]'])
plt.title("Microwave Usage spread during day", fontsize=20, fontweight='bold')


# In[ ]:


# Findings:

# Microwave usage during the day stays at a low steady pace most of the time.
# This might be because the microwave is plugged in at all times and is using a small amount of 
# electricty for the display lights etc
# There are some peaks in the electrcity usage during the day indicating usage of the microwave to 
# heat up food (which is not very often but only at meal times).


# In[436]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=night_scatter['Date'], y=night_scatter['Microwave (R) [kW]'])
plt.title("Microwave Usage spread during night", fontsize=20, fontweight='bold')


# In[437]:


# Findings:

# However, during the night, the microwave only requires a little steady electricity while being plugged into 
# the power. There are no fluctuations or peaks during the night time meaning no one wants to heat food at night.
# There is one anomoly in the night scatter plot, which might be an error while gathering data.


# In[438]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=day_scatter['Date'], y=day_scatter['Fridge (R) [kW]'])
plt.title("Fridge Usage spread during day", fontsize=20, fontweight='bold')


# In[439]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=night_scatter['Date'], y=night_scatter['Fridge (R) [kW]'])
plt.title("Fridge Usage spread during night", fontsize=20, fontweight='bold')


# In[66]:


# Findings:

# Fridge mostly has the same spread of data during the night and day.
# The fridge is plugged into power at all times and uses about the same electricty to keep a low temperature
# inside, irrespective of the time of the day.
# However, generally the usage is less during the night (3 kW) compared to day (4kW) because maybe
# people open the fridge door more often during the day and the fridge reqires more electricty to maintain
# the cool temperature unlike when poeple are asleep at night and no one opens the firdge door and therefore no 
# cool air is lost.


# In both the scatter plots for the microwave and fridge, there is zero usage from 2014-09 to 2014-11, 
# This can be due to multiple reasons for example stop gathering data, or damage to the device that 
# is recording the voltage use etc.


# In[ ]:




