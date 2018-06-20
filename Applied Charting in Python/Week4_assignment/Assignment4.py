
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **weather phenomena** (see below) for the region of **Roodepoort, Gauteng, South Africa**, or **South Africa** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Roodepoort, Gauteng, South Africa** to Ann Arbor, USA. In that case at least one source file must be about **Roodepoort, Gauteng, South Africa**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Roodepoort, Gauteng, South Africa** and **weather phenomena**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **weather phenomena**?  For this category you might want to consider seasonal changes, natural disasters, or historical trends.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# # 1. A brief analysis of cryptocurrencies
# 
# Since the launch of Bitcoin in 2008, hundreds of similar projects based on the blockchain technology have emerged. We call these cryptocurrencies (also coins or cryptos in the Internet slang). Some are extremely valuable nowadays, and others may have the potential to become extremely valuable in the future1. 
# 
# This analysis aims to highlight some of the largest cryptocurrencies by market capitalisation, as well as to highlight some risks of some crypocurrencies by displaying their price volatility.

# In[12]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
plt.style.use('fivethirtyeight')

# Reading in current data from coinmarketcap.com
current = pd.read_json("https://api.coinmarketcap.com/v1/ticker/")

# Printing out the first few lines
current.head(3)


# # 2. Full dataset, filtering, and reproducibility
# 
# Analysis cannot be reproducible with live online data. To solve these problems, CSV has been saved based on data loaded on the 6th of December of 2017 using the API call https://api.coinmarketcap.com/v1/ticker/?limit=0 named coinmarketcap_06122017.csv  in the notebook.

# In[13]:

# Reading coinmarketcap_06122017.csv into pandas
dec6 = pd.read_csv('coinmarketcap_06122017.csv')

# Selecting the 'id' and the 'market_cap_usd' columns
market_cap_raw = dec6[['id', 'market_cap_usd']]

# Counting the number of values
market_cap_raw.count()


# # 3. Discard the cryptocurrencies without a market capitalization
# 
# Performed the above count() for id and market_cap_usd differ to identify cryptocurrencies that have no known market capitalization, this is represented by NaN in the data, and NaNs are not counted by count(). These cryptocurrencies are removed below.

# In[22]:

# Filtering out rows without a market capitalization
cap = market_cap_raw.query('market_cap_usd > 0')

# Counting the number of values again
cap.count()


# # 4. How big is Bitcoin compared with the rest of the cryptocurrencies?
# 
# Bitcoin is under serious competition from other projects, but it is still dominant in market capitalization. The graph below plots the market capitalization for the top 10 coins as a barplot to better visualize this.

# In[25]:

#Declaring these now for later use in the plots
TOP_CAP_TITLE = 'Top 10 market capitalization'
TOP_CAP_YLABEL = '% of total cap'

# Selecting the first 10 rows and setting the index
cap10 = cap[:10]
cap10.set_index('id', inplace=True)

# Calculating market_cap_perc
cap10['market_cap_perc'] = cap10.iloc[:,:].apply(lambda x: (x / cap.market_cap_usd.sum()) * 100)

# Plotting the barplot with the title defined above 
ax = cap10['market_cap_perc'].plot.bar()
plt.title(TOP_CAP_TITLE)
plt.ylabel(TOP_CAP_YLABEL )
plt.xlabel('Crypto Currency')
ax.set_xticklabels(cap10.index.values, rotation=45 )


# # 5. Making the plot easier to read 
# 
# Bitcoin is too big, and the other coins are hard to distinguish because of this. Instead of the percentage, the graph below uses a log10 scale of the capitalization. 

# In[27]:

# Plotting market_cap_usd as before but adding the colors and scaling the y-axis  
ax = cap[:10]['market_cap_usd'].plot.bar(colors = 'grey', logy=True)
plt.ylabel('USD')
ax.set_xticklabels(cap10.index.values, rotation=45 )
plt.title(TOP_CAP_TITLE)
plt.xlabel('')
plt.show()


# # 6. Analysing volatility in cryptocurrencies
# 
# The cryptocurrencies market has been spectacularly volatile since the first exchange opened. The section below selects and prints the 24 hours and 7 days percentage change.

# In[29]:

# Selecting the id, percent_change_24h and percent_change_7d columns
volatility = dec6[['id', 'percent_change_24h','percent_change_7d']]

# Setting the index to 'id' and dropping all NaN rows
volatility.set_index('id', inplace=True)
volatility = volatility.dropna()

# Sorting the DataFrame by percent_change_24h in ascending order
volatility = volatility.sort(['percent_change_24h'], ascending=[1])

# Checking the first few rows
print(volatility.head())


# # 7. Observing volatility
# 
# Some cryptocurrencies experience large price swings. The graph below plots the top 10 biggest gainers and top 10 losers in market capitalization.

# In[32]:

#Defining a function with 2 parameters, the series to plot and the title
def top10_subplot(volatility_series, title):
    # Making the subplot and the figure for two side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    # Plotting with pandas the barchart for the top 10 losers
    volatility_series[:10].plot.bar(ax=axes[0], color = 'darkred')
    
    # Setting the figure's main title to the text passed as parameter
    fig.suptitle(title)
    
    # Setting the ylabel to '% change'
    axes[0].set_ylabel('% change')
    axes[0].set_xlabel('', rotation = 45)
    
    # Same as above, but for the top 10 winners
    volatility_series[-10:].plot.bar(ax=axes[1], color = 'darkblue')
    axes[1].set_xlabel('')
    
    # Returning this for good practice, might use later
    return fig, ax

DTITLE = "24 hours top losers and winners"

# Calling the function above with the 24 hours period series and title DTITLE  
fig, ax = top10_subplot(volatility['percent_change_24h'], DTITLE)


# # 8. Checking the weekly series
# Reusing the function defined above to see what is going weekly instead of daily.

# In[33]:

# Sorting in ascending order
volatility7d = volatility.sort(['percent_change_7d'], ascending=[1])
WTITLE = "Weekly top losers and winners"

# Calling the top10_subplot function
fig, ax = top10_subplot(volatility7d['percent_change_7d'], WTITLE)


# # 9. Analysing smaller cryptocurrencies?
# 
# Smaller cryptocurrencies seem to be less stable projects in general, and therefore even riskier investments than the bigger ones. The section below checks how many projects are large currencies with significant market capitalization.

# In[34]:

# Selecting everything bigger than 10 billion 
largecaps = cap[cap['market_cap_usd'] >= 10000000000]

# Printing out largecaps
print(largecaps)


# # 10. Plotting coins of various market caps together
# 
# The section below plots count of cryptocurrencies into various categories of small to large market cap coins.

# In[38]:

# Function for counting different marketcaps from the "cap" DataFrame. Returns an int.
def capcount(query_string):
    return cap.query(query_string).count().id

# Labels for the plot
LABELS = ["large", "micro", "nano"]

# Using capcount count the large cryptos
large = capcount('300000000 <= market_cap_usd')

# Same as above for micro ...
micro = capcount('50000000 <= market_cap_usd < 300000000')

# ... and for nano
nano =  capcount('market_cap_usd < 50000000')

# Making a list with the 3 counts
values = [large, micro, nano]

# Plotting them with matplotlib 
plt.bar(range(len(values)), values, tick_label=LABELS)
plt.title('Count of cryptocurrencies based on market cap')

