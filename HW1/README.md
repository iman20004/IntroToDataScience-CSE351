## CSE 351 - Introduction to Data Science Assignment 1: Exploratory Data Analysis

### Data
The assignment is based on New York City Airbnb Open Data [2]. The main task is to mine the data and uncover interesting observation about the different hosts and areas.

### Tasks
1. Examine the data, there may be some anomalies in the data, and you will have to clean the data before you move forward to other tasks. Explain what you did to clean the data. (10 Points)
2. Examine how the prices of the Airbnb changes with the change in the neighborhood.
  a. Find Top 5 and Bottom 5 neighborhood based on the price of the Airbnb in that neighborhood (select only neighborhoods with more than 5 listings). (10 Points)
  b. Analyze, the price variation between different neighborhood group, and plot these trends. (5 Points)
3. Select a set of the most interesting features. Do a pairwise Pearson correlation analysis on all pairs of these variables. Show the result with a heat map and find out most positive and negative correlations. (5 points)
4. The Latitude and Longitude of all the Airbnb listings are provided in the dataset.
   a. Plot a scatter plot based on these coordinates, where the points represent the location of an Airbnb, and the points are color coded based on the neighborhood group feature. (5 Points)
   b. Now again, plot a scatter plot based on these coordinates, where the points represent the location of an Airbnb, and the points are color coded based on the price of the particular Airbnb, where price of the listing is less than 1000. Looking at the graph can you tell which neighborhood group is the most expensive. (5 Points)
5. Word clouds are useful tool to explore the text data. Extract the words from the name of the Airbnb
and generate a word cloud. (5 Points)
6. Find out which areas has the busiest (hosts with high number of listings) host? Are there any reasons, why these hosts are the busiest, considers factors such as availability, price, review, etc.? Bolster you reasoning with different plots and correlations. (10 Points)
7. Create two plots (at least one unique plot not used above) of your own using the dataset that you think reveals something very interesting. Explain what it is, and anything else you learned. (10 Points)
8. Visual Appeal and Layout - For all the tasks above, please include an explanation wherever asked and make sure that your procedure is documented (suitable comments) as well as you can. Don’t forget to label all plots and include legends wherever necessary as this is key to making good visualizations! Ensure that the plots are visible enough by playing with size parameters. Be sure to use appropriate color schemes wherever possible to maximize the ease of understandability. Everything must be laid out in a python notebook(.ipynb). (5 Points)
