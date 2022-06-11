#!/usr/bin/env python
# coding: utf-8

# # CSE351 HW1: Exploratory Data Analysis
# ## NYC Airbnb 2019
# **Iman Ali (112204305)**

# In[14]:


# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud


# ## Load the data
# Data source: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

# In[15]:


data = pd.read_csv('AB_NYC_2019.csv') # use pandas to load the data
data.head() # display first 5 rows of the dataset


# 
# ## Task 1 (Examine & Clean data)

# Examine. First get a high level understanding of the data.

# In[16]:


# A concise summary of the DataFrame. Check which properties are being represented and what their types are.
data.info()


# In[17]:


# Generate descriptive statistics for all integer/float type properties.
data[['price', 'minimum_nights', 'number_of_reviews','reviews_per_month', 
     'calculated_host_listings_count', 'availability_365']].describe()


# In[18]:


# Boroughs being represented 
pd.unique(data['neighbourhood_group'])


# In[19]:


# Listings in each borough
plt.figure(figsize=(12,4))
sns.countplot(x="neighbourhood_group", data=data)
plt.title("Number of listings in each borough")


# In[20]:


# Average price in each borough
grouped_data = data.groupby("neighbourhood_group")
grouped_data['price'].mean()


# In[21]:


# Number of hosts
len(pd.unique(data['host_id']))


# In[22]:


# Types of rooms
pd.unique(data['room_type'])


# In[23]:


# Neighbourhood represented
pd.unique(data['neighbourhood'])


# Now we will clean the data by checking for outliers, missing values, abnormal values etc

# ## Data uniqueness and duplicates

# In[24]:


# There are 48895 data entries. Ensure that each listing has a unique id.
data["id"].is_unique


# In[25]:


# Confirm there are no duplicates in the data set
data.duplicated().any()


# ## Missing/Null Values

# In[26]:


# Check missing values in the data.
data.isna().sum()


# In[27]:


# Entries with name, host name, or last_review fields missing (NaN) do not corrupt the data 
# because they are object types.
# However reviews_per_month is a float type and should not be NaN


# Analyze listings with missing reviews_per_month field
data[data.reviews_per_month.isna()][['id','name','reviews_per_month', 'number_of_reviews']]


# In[28]:


# We can confirm that all the listings where reviews_per_month field is NaN, have zero number_of_reviews.
# Therefore we can safely replace the NaN reviews_per_month to zero

data['reviews_per_month'] = data['reviews_per_month'].fillna(value=0)


# In[29]:


# Certify reviews_per_month column NaN values have been replaced to zero

data.loc[data.number_of_reviews == 0, ['id','name','reviews_per_month', 'number_of_reviews']]


# ## Anomalies in Price (Detecting & treating outliers)

# In[30]:


# General stats about the price
data['price'].describe()


# In[31]:


# Check the spread(range) of prices for outliers
plt.figure(figsize=(15,7))
sns.scatterplot(data=data, x="id", y="price", hue="room_type")
plt.title("Price Range")


# In[32]:


# Delete any listings where price is zero or negative (Eleven listings)
len(data[data.price == 0])


# In[33]:


data.drop(data[data.price <= 0].index, inplace=True)
len(data[data.price == 0])


# In[34]:


# Scatter plot shows data is quite evenly spread with most of the listings under $6000


# Analyze rooms with higher price tag (>= $6000) according to room type
plt.figure(figsize=(16,6))
sns.boxplot(data=data, x="price", y="room_type")
plt.title("Price vs Room type")


# In[35]:


# Analyze rooms with higher price tag (>= $6000) according to area of listing (neighbourhood/borough)
plt.figure(figsize=(16,6))
sns.boxplot(data=data, x="price", y="neighbourhood_group")
plt.title("Price vs Borough")


# In[36]:


# Most of the rooms with prices greater than 6000 are almost always entire home/apt in Manhattan or Brooklyn
# therefore justifying the high price tag
data[data.price >= 6000]


# In[37]:


# However, the box plot "Price vs Borough" does show an abormal outlier in Queens. 
# This outlier is priced at $10,000 (way more than other rooms in the area) 
# Further this is a 'private room' type with 0 availability and 100 day minimum stay.
# To me this seems like an error, therefore I will drop it.
pricey_rooms = data[data.price == 10000]
queens_outlier = pricey_rooms[pricey_rooms.neighbourhood_group=="Queens"]
data.drop(queens_outlier.index, inplace=True)

# However, replacing outliers with a median/mean value is a better approach.
# This method of treating outliers is known as mean/median imputation


# ## Anomalies in Availability

# In[38]:


# General stats about the availability
data['availability_365'].describe()


# In[39]:


# Spread of data seems good with less availability in Manhattan and Brooklyn, 
# While Staten Island seems to have more rooms that go unoccupied

plt.figure(figsize=(15,7))
sns.boxplot(x=data.availability_365, y=data.neighbourhood_group)
plt.title("Availability in each Borough")


# ## Anomalies in Minimum Nights

# In[40]:


# General stats about the min nights
data['minimum_nights'].describe()


# In[41]:


# Check for outliers in scatterplot
plt.figure(figsize=(15,7))
sns.scatterplot(x=data.id, y=data.minimum_nights)
plt.title("Minimum nights data spread")


# In[42]:


# Some outliers with greater than 800 nights minimum stay
# However, they are very few in number and I will not delete them.
# They seem understandable due to their low prices and high availabilty rate.
data[data.minimum_nights > 800]


# ## Anomalies in Reviews per Month

# In[43]:


# General stats about reviews_per_month
data['reviews_per_month'].describe()


# In[44]:


# Check for outliers in scatterplot
plt.figure(figsize=(15,7))
sns.scatterplot(x=data.id, y=data.reviews_per_month)
plt.title("Reviews per month data spread")


# In[45]:


# One outlier found in the scatter plot
data[data.reviews_per_month > 40]


# In[46]:


# This seems like an error to me because this outlier has way more than average reviews
# Yet it has low occupancy rate with the room available 299 out of 365 days.
# That means this listing got guests for a total of 60 nights in the whole year
# So it cannot be getting 60 reviews per month


# I will use the mean imputation method to treat this outlier and replace its value with the average.
data.loc[data.reviews_per_month > 40, 'reviews_per_month'] = data['reviews_per_month'].mean()


# In[47]:


# General stats after the outlier has been removed.
data['reviews_per_month'].describe()


# ## Task 2 Price vs Neighbourhood

# ### Part a) Find Top 5 and Bottom 5 neighborhood based on the price of the Airbnb

# In[48]:


# First get data entries of only those neighbourhoods with more than 5 listings 
neighs = data[data.groupby('neighbourhood')['neighbourhood'].transform('size') > 5]
neighs


# In[49]:


# Initially there were 221 neighbourhoods. 
len(data.groupby('neighbourhood'))


# In[50]:


# After removing the ones with <=5 listings we have 190 neighbourhoods left
len(neighs.groupby('neighbourhood'))


# In[51]:


# Check the spread of data and detect outliers
plt.figure(figsize=(18,8))
sns.scatterplot(x=neighs.neighbourhood, y=neighs.price)
plt.title("Price vs Neighbourhood")


# In[52]:


# Data has a skewed distribution and there are some outliers as seen above
# Therefore, get median statistics of each neighbourhood 
# Since median will be a better central tendancy measure here

stats=neighs.groupby('neighbourhood').median()
stats[['price', 'number_of_reviews', 'availability_365']]


# In[53]:


# Visualize the median price in each neighbourhood
graph=stats.plot.bar(y='price', figsize=(35,15))
plt.title("Median Price in each Neighbourhood")


# In[54]:


# Top 5 neighbourhoods with highest median listings
top=stats.sort_values('price').tail(5)[["price"]]
top[::-1]


# In[55]:


# Bottom 5 neighbourhoods with highest median listings
stats.sort_values('price').head(5)[["price"]]


# ### Part b) Price variation between different neighbourhood group

# In[56]:


# Plotting the trend between price and neighborhood groups
plt.figure(figsize=(15,8))
sns.barplot(x="neighbourhood_group", y="price", data=data)
plt.title("Prices vs Neighborhood groups")


# In[57]:


# Mean and median price trends also shown
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,5))
data.groupby('neighbourhood_group').median().plot.bar(y='price', title=("Median prices in each Borough"), ax=ax1)
data.groupby('neighbourhood_group').mean().plot.bar(y='price', title=("Mean prices in each Borough"), ax=ax2)


# In[58]:


# Standard deviation in prices between different neighbourhood groups
data.groupby('neighbourhood_group').std().plot.bar(y='price', figsize=(8,5), title=("Standard deviation in prices of each Borough"))


# ## Task 3 Pearson correlation analysis

# In[59]:


# Properties represented in the data frame:
data.dtypes


# In[60]:


# id, name, host_id, host_name do not have any predictive power or effect on the other values
# longitude, latitude are values that describe the location of the listing
# However, neighbourhood_group is a better measure of location and is related to other properties more closely
# last_review is another property which should not have any effect on the others because it is just a date object
# Reviews can be described as the number of reviews received in total or monthly average. 

# Therefore I will ignore those properties and consider the remaining, which are:
# neighbourhood_group, room_type, price, minimum_nights, number_of_reviews, reviews_per_month, 
# calculated_host_listings_count, availability_365                               



# But first we need to convert the neighbourhood_group and room_type properties to numeric values.
# This is done so a heatmap can be created with these properties. (we need numeric values)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7))

# Listings in each borough
sns.countplot(x="neighbourhood_group", 
              data=data,
              order = data['neighbourhood_group'].value_counts().index,
              ax=ax1).set_title("Number of listings in each Neighbourhood group")

# Number of each room type
sns.countplot(x="room_type", 
              data=data, 
              order = data['room_type'].value_counts().index,
              ax=ax2).set_title("Room type frequency")
plt.show()


# In[61]:


# From the first plot "Number of listings in each Neighbourhood group", we can deduce that the order of frequency is:
# Manhattan (5), Brooklyn (4), Queens (3), Bronx (2), Staten Island (1) in descreasing order 

heat_data = data.copy()

# Neighbourhood_group column mapped to integers based on number of listings in each
heat_data['neighbourhood_group'] = heat_data['neighbourhood_group'].map({'Manhattan':5,'Brooklyn':4,
                                                                         'Queens':3,'Bronx':2,
                                                                         'Staten Island':1})

# From the second plot "Room type frequency", we can deduce the order of frequency of rooms is:
# Entire home/apt (2), Private room (2), Shared room (1) in decreasing order
# Room_type column mapped to ints based on frequency of each type
heat_data['room_type'] = heat_data['room_type'].map({'Entire home/apt':3,
                                                     'Private room':2,
                                                     'Shared room':1})


# In[62]:


# Generate the heat map 
mapdata = heat_data[['neighbourhood_group', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 
                     'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]                                
corr = mapdata.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, vmin=-1)
plt.title("Heatmap")


# In[63]:


# The most positive correlation (0.56) is between number_of_reviews and reviews_per_month.
# This is a very expected result because the greater number of reviews a listing receives in a month,
# the greater the total number of reviews it receives in total.

# The second most positive correlation (0.25) is between room_type and price. 
# This is again very understandable because entire home/apt cost greater than private rooms, 
# and private rooms cost greater than shared rooms.

# The most negative correlation (-0.12) is between minimum_nights and reviews_per_month.
# This could entail that guests do not like a high minimum night requirement.
# Listing that require the guests to book for more nights get less reviews since customers dont want to be
# forced to book for multiple nights (they might have come for a short weekend trip)

# The second most negative correlation (-0.11) is between neighbourhood_group and reviews_per_month.
# This is an interesting relation. It might be because popular boroughs like manhattan and brooklyn get busy guests,
# or people that travel often and dont leave many reviews, while staten island might get more family oriented guests.


# ## Task 4 Latitude and Longitude

# ### Part a. Scatter plot (points represent location and are color coded according to neighbourhood group)

# In[64]:


plt.figure(figsize=(15,12))
sns.scatterplot(x=data.longitude, y=data.latitude, hue=data.neighbourhood_group)
plt.title("Location Scatter Plot")


# ### Part b. Scatter plot (points represent location and are color coded according to price)

# In[65]:


price_data = data[data.price <1000]
price_data = price_data.sort_values(by="price")

plt.figure(figsize=(15,12))
sns.scatterplot(data=price_data, x="longitude", y="latitude", hue="price")
plt.title("Location Scatter Plot (Price color coded)")


# #### We can easily tell just by looking at both the plots that Manhattan is the most expensive neighbourhood group.

# ## Task 5 Word Cloud

# In[66]:


# Get names of all the airbnb listings and convert them into a big string
airbnb_names = data['name'].agg(lambda x: ','.join(map(str, x)))


# In[67]:


# Create the word cloud
cloud = WordCloud(width = 800, height = 600).generate(airbnb_names)

# Display the generated image:
plt.figure(figsize=[16,12])
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# The cloud shows the frequently appearing words. Location names like Manhattan, Brooklyn, East Village, Williamsburg are common to see in listing names. 'Private room' (similarly 'studio', 'private bedroom' or 'cozy room') are other popular words. This seems appropriate because people prefer to get rooms instead of entire apartments due to the high prices in NYC. Further, we can spot some adjectives like spacious, quiet, large, charming, luxurious etc. Descriptive words are used in listing names to draw customers. It lets the guest know what to expect while highlighting the key features of the property. Lastly tourist attractions like central park and times square are also included in listing names.

# ## Task 6 Busiest Hosts

# ### Which areas have the busiest hosts

# In[68]:


# Get the top 20 busiest hosts (hosts with the highest number of listings in a neighbourhood group)

# Group first by host_id, and then neighbourhood_group
# host_name is just added to display the name of the host 
# host_id and host_name will always have one same group (since they are unique)
hosts = data.groupby(["host_id", "host_name", "neighbourhood_group"])

# counting the size() will give the number of listings of each host in an area
# get the top 20
top_hosts = hosts.size().nlargest(20) 
top_hosts


# These are the TOP 20 BUSIEST HOSTS
# And it can be clearly seen that Manhattan is the area with the most busy hosts


# In[69]:


# The top 10 busiest hosts are:
# 1. Sonder (NYC) -- 327 listings in Manhattan
# 2. Blueground -- 230 listings in Manhattan
# 3. Kara -- 121 listings in Manhattan
# 4. Sonder -- 96 listings in Manhattan
# 5. Jeremy & Laura -- 96 listings in Manhattan
# 6. Corporate Housing -- 91 listings in Manhattan
# 7. Ken -- 86 listings in Manhattan
# 8. Kazuya -- 79 listings in Queens
# 9. Pranjal -- 65 listings in Manhattan
# 10. Mike -- 52 listings in Manhattan


# In[70]:


# Get data (all listings) of the top 10 hosts
top_hosts_id = [219517861, 107434423, 30283594, 12243051, 16098958, 61391963, 
                22541573, 137358866, 200380610, 1475015]
top_hosts_data = data.loc[data['host_id'].isin(top_hosts_id)]
top_hosts_data


# In[71]:


plt.figure(figsize=(12,4))
sns.barplot(x="host_name", 
            y="calculated_host_listings_count", 
            data=top_hosts_data.sort_values('calculated_host_listings_count', ascending=False))
plt.title("Top 10 busiest hosts & their listings count")


# In[72]:


# This figure again reiterates that most of the busy hosts have listings in Manhattan
plt.figure(figsize=(12,4))
sns.countplot(x="neighbourhood_group", data=top_hosts_data)
plt.title("Number of busy host listings in different areas")


# In[73]:


# Task 6 asks us to 'Find out which AREAS have the busiest host?'
# We have deduced that Manhattan is the area (neighbourhood_group) with the most busy hosts
# But we can further continue our investigation to find the neighbourhoods with the most busy hosts

data.groupby(["host_id", "host_name", "neighbourhood_group", "neighbourhood"]).size().nlargest(20)


# In[74]:


# The list above shows which neighbourhoods have listings from the most busy hosts
# The plot below helps us visualize that data for the top 10 neighbourhoods

plt.figure(figsize=(15,5))
sns.countplot(x="neighbourhood", 
              data=top_hosts_data, 
              order=top_hosts_data.neighbourhood.value_counts().iloc[:10].index)
plt.title("Top 10 neighbourhoods with the most busy hosts")


# ### Why are these hosts the busiests
# ##### Considers factors such as availability, price, review

# In[75]:


# Availability of busiest hosts' listings

# Majority of these listings have a high availability (>200) as the graph demonstrates 
# This means that most of these airbnbs are not occupied.
# Since these hosts are so busy in terms of having numerous properties (for exmample 1 host has 327 listings)
# it might be hard for them to update their rooms or add extra details to attract more customers.
# Another reason for high availability could be the fact that most of these listing are in 
# Manhattan which generally has a very high number of options available including hotels.

plt.figure(figsize=(12,4))
sns.barplot(x="host_name", y="availability_365", data=top_hosts_data)
plt.title("Availability of busy hosts listings")


# In[76]:


# Price of busiest hosts' listings

# Most of these listings have a higher price with the exception of Kazuya's listings
# This is because Kazuya has listings in Queens, while all the others are located in Manhattan
# Manhattan is generally the most expensive out the 5 boroughs of New York.

# However, still most of these listings are under $300 per night which is a very reasonable rate 
# for a property in Manhattan.

plt.figure(figsize=(12,4))
sns.barplot(x="host_name", y="price", data=top_hosts_data)
plt.title("Prices of busy hosts listings")


# In[77]:


# Reasonable prices are popular in Manhattan.
# People want to save money since vacationing in Manhattan is already expensive,
# yet they do not want to live in shared rooms or extremely cheap places.
# Therefore, most people settle for the middle where they can get affordable luxury.

top_hosts_data.describe()["price"]


# In[78]:


# Reviews of busiest hosts' listings

# Number of reviews are generally low for these listings (less than 5)
# This might be because these listings are not out of the ordinary. 
# The busy hosts have a lot of listings to manage and do not put the extra effort to 
# request guests for reviews or add extra special features to which guests would comment 
# and recommend.


plt.figure(figsize=(12,4))
sns.barplot(x="host_name", y="number_of_reviews", data=top_hosts_data)
plt.title("Number of reviews of busy hosts listings")


# In[79]:


# Room type of busiest hosts' listings

# Most of the listings of the busy hosts are entire home/apt.
# New york is one of the busiest places and top tourist destinations in the world.
# People usually travel with either their family or friends in large groups.
# That is why customers prefer to get an entire apartment so they can all stay together
# as compared to getting hotel rooms
# Customers also appreciate the privacy they get with the entire property and no sharing with strangers.

# Further, we already know from our analysis that entire home/apt listings are more expensive
# These busy hosts can charge more for their listings as compared to single rooms.

plt.figure(figsize=(12,4))
sns.countplot(x="room_type", data=top_hosts_data)
plt.title("Room type of busy hosts listings")


# In[80]:


# Minimum nights of busiest hosts' listings

# Almost all the busy hosts have a common threshold requirement for minimum number of nights that can be booked
# All require around 30 nights at least. This means that these listings are 1 month rentals
# Therefore, the busy hosts only need to find one customer/guest in one month and then not worry about their listing
# This makes it easy to handle multiple properties.

plt.figure(figsize=(12,4))
sns.barplot(x="host_name", y="minimum_nights", data=top_hosts_data)
plt.title("Minimum nights requirement of busy hosts listings")


# In[81]:


# Correlation of other factors (heat map drawn similar to Task 3)

corr_data = top_hosts_data.copy()
corr_data['neighbourhood_group'] = corr_data['neighbourhood_group'].map({'Manhattan':5, 'Brooklyn':4,'Queens':3, 
                                                                         'Bronx':2,'Staten Island':1})
corr_data['room_type'] = corr_data['room_type'].map({'Entire home/apt':3,'Private room':2,'Shared room':1})

heat_data = corr_data[['neighbourhood_group', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 
                     'reviews_per_month', 'calculated_host_listings_count', 'availability_365']] 
corr = heat_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, vmin=-1)
plt.title("Heatmap for top hosts")


# In[82]:


# Interesting observations from heatmap

# High positive correlation b/w neighbourhood_group and room_type (0.86)
# This is expected because different areas in NYC, get different types of travellers
# For example the financial district in Manhattan with career oriented individuals who want entire apt 
# vs Williamsburg with younger generation or artists etc who would prefer shared rooms or private rooms

# Highly negative correlation b/w minimum_nights and reviews_per_month (-0.84)
# The greater the number of nights required to book the listings, the less the number of reviews
# which as discussed above is understandble because people don't want to be forced to book a lot of nights

# Highly positive correlation b/w reviews_per_month and calculated_host_listings_count (0.78)
# The more the number of listings the host has, the more loyal customers it has that leave more reviews
# The busy hosts also have more experience and might spend more time and money to advertise their many listings
# and ask their guests for reviews


# ## Task 7 (Two interesting plots)

# ### Interesting find #1

# In[83]:


# Analyze what percentage is each type of room (entire home/apt, private or shared room)

# The pie chart below is a great visual proof of the fact that almost 52% of the listings are 
# offering properties that are booked as Entire home/apartments

labels = data['room_type'].value_counts().index
rooms = data['room_type'].value_counts().values
plt.figure(figsize=(10,10))
plt.pie(rooms, labels=labels, autopct='%1.1f%%', explode=[0.03]*3, pctdistance=0.5, textprops={'fontsize': 16})
plt.title("Room type percentage")


# In[84]:


# We can see that entire home/apt listings have the highest average price
# While shared rooms have the lowest average price

plt.figure(figsize=(8,6))
type_price= data.groupby('room_type', as_index=False)[['price']].mean()
sns.barplot(x=type_price.room_type, y=type_price.price)
plt.title("Mean prices of room types");


# In[85]:


# We have concluded that the room type entire home/apt is the most frequent and the most expensive
# Therefore we can deduce that most of the income generated through the airbnb industry in NYC in 2019
# came from entire home/apt listings

labels = data['room_type'].value_counts().index
roomtype_share =  data.groupby("room_type")['price'].sum()
plt.figure(figsize=(10,10))
plt.pie(roomtype_share, labels=labels, autopct='%1.1f%%', explode=[0.03]*3, pctdistance=0.5, textprops={'fontsize': 16})
plt.title("Room_type Price Share")


# The graph below shows that 72% of the price share of the airbnb industry in 2019 came from entire home/apt listings


# In[86]:


# Similarly Analyze the percentage income generated by each neighbourhood group

# The piechart below shows that 57% of the total price is being generated by Manhattan 
# then Brooklyn, then Queens, then Bronx and then Staten Island

labels = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
area_share =  data.groupby("neighbourhood_group")['price'].sum()
plt.figure(figsize=(10,10))
plt.pie(area_share, labels=labels, autopct='%1.1f%%', explode=[0.05]*5, pctdistance=0.5, textprops={'fontsize': 12})
plt.title("Neighbourhood Price Share")


# ### Interesting find #2

# In[87]:


# Here we analyze the availability of each room type in each neighbourhood 

plt.figure(figsize=(12,8))
sns.lineplot(x=data.neighbourhood_group, y=data.availability_365, hue=data.room_type)
plt.title('Availability of each room type in each neighbourhood')


# In[88]:


# Some interesting factors observed in above graph:

# Staten Island has the least availability of shared rooms, and 
# the highest availabilty of entire home/apt and private rooms.

# Staten Island is mostly a residential borough with properties like houses instead of 
# apartment complexes or buildings.
# Therefore, hosts usually either rent out their entire place or a private room in their house.
# However, Staten Island is not a big tourist destination and that is why people prefer to rather
# stay in Manhattan or Brooklyn. This explains the high availabilty rate of entire home/apt, private rooms
# in Staten Island who gets less customers. 
# On the other hand, there arent many shared room listings in staten island (only 9 listings)
# Since there are only 9 listings available, and we know shared rooms are cheaper,
# they have the least availability (maybe because they are always occupied or maybe 
# because the host doesn't want to rent a room in their house too often)

len(data.loc[(data.neighbourhood_group == 'Staten Island') & (data.room_type == 'Shared room')])


# In[89]:


# Another interesting observation is that Manhattan, Brooklyn and Queens
# have the highest availabilty for shared rooms.
# This is quite understandable because most people (guests) are not ready to compromise
# on their privacy.
# Shared rooms are usually booked by a certain niche such as college students or people travelling
# on a small budget who dont mind sharing the room with a stranger.
# Shared rooms also compromise security because you need to be able to trust the stranger to leave your
# belongings and luggage in the room with them.



# Also we can see that Brooklyn and Manhattan have very similar availabilites.
# This goes to show that these two neighbourhoods are the most popular, and have the same trend
# with least availabilty of rentals in general. 


# In[90]:


# Availabilty of room types


# The figure below has a uniform distribution, and it shows that rooms of every type and every price  
# range are available throughout.

plt.figure(figsize=(12,8))
sns.scatterplot(data=data[data.price < 300], x="availability_365", y='price', hue='room_type')
plt.title("Availabilty of room types of different price range")


# In[91]:


# Minimum night requirement in each neighbourhood group

plt.figure(figsize=(12,8))
sns.lineplot(x=data.neighbourhood_group, y=data.minimum_nights)
plt.title("Minimum night requirement in each borough")


# In[92]:


# This graph shows that Manhattan, Brooklyn and Queens generally have a tight range of
# minimum night requirement in all its listings. Staten Island and the Bronx have a wider
# range of requirements available, meaning they are more flexible.

# Manhattan has the highest minimum night requirement. Brooklyn and Queens have similar
# requirements and Bronx has the least.
# However, it can be noted that generally all the borough have a low minimum night requirement (<9)

