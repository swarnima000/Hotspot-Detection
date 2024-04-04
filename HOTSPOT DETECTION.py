#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


get_ipython().system('pip install basemap')
get_ipython().system('pip install basemap-data')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score as silhouette
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


df = pd.read_csv('covid-19-all.csv', low_memory=False)
df.head()


# # EDA and Pre-processing

# In[5]:


df.isnull().sum()


# In[6]:


null_labels = ['Country/Region', 'Province/State', 'Latitude', 'Longitude', 'Confirmed', 'Recovered', 'Deaths', 'Date']
null_explode = [0.05, 0.05, 0.05, 0.05, 0.5, 0.4, 0.3, 0.2]
plt.pie(null_vals,labels = null_labels, autopct = '%1.2f%%', explode = null_explode, startangle = 90)
plt.legend(bbox_to_anchor=(1,1.2))
plt.show()


# In[7]:


df.shape


# In[8]:


df.tail()


# In[9]:


df = df.dropna(subset = ['Latitude', 'Longitude', 'Confirmed'])


# In[10]:


df.isnull().sum()


# In[11]:


df.head()


# In[12]:


df.Date.unique()


# In[13]:


#date: '2020-12-31'
data = df.loc[(df.Date == '2020-08-26')]
data


# In[14]:


data = data.drop(['Country/Region', 'Province/State', 'Recovered', 'Deaths', 'Date'], axis = 1)


# In[15]:


data


# # Unsupervised Clustering Algorithms

# ## K-Means Algorithm

# In[16]:


# Train the KMeans clustering algorithm
kmeans = KMeans(n_init = 42, n_clusters=8)
labels = kmeans.fit_predict(data)


# In[17]:


labels


# In[18]:


# Calculate the silhouette score
silhouette_score = silhouette(data, labels, metric='euclidean')

print("Silhouette Score:", silhouette_score)


# In[20]:


# Plot the data points with different colors for each cluster
m = Basemap(projection='merc',
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=20,
            resolution='c')

latitudes = data.loc[:, 'Latitude']
longitudes = data.loc[:, 'Longitude']
cases = data.loc[:, 'Confirmed']
x, y = m(longitudes, latitudes)
m.drawcoastlines()
m.drawcountries()
m.scatter(x, y, s=cases*0.005, c=labels, cmap='viridis', alpha=0.5)

plt.show()


# ## Birch Algorithm 

# In[21]:


# fit the birch clustering model to the dataset
birch = Birch(threshold=0.5, branching_factor=50)
birch_model = birch.fit(data)


# In[22]:


# predict the clusters
labels = birch_model.predict(data)
normal_labels1 = labels.tolist()


# In[23]:


# Calculate the silhouette score
silhouette_score = silhouette(data, normal_labels1, metric='euclidean')

print("Silhouette Score:", silhouette_score)


# In[24]:


# Plot the data points with different colors for each cluster
m = Basemap(projection='merc',
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=20,
            resolution='c')

latitudes = data.loc[:, 'Latitude']
longitudes = data.loc[:, 'Longitude']
cases = data.loc[:, 'Confirmed']
x, y = m(longitudes, latitudes)
m.drawcoastlines()
m.drawcountries()
m.scatter(x, y, s=cases*0.005, c=labels, cmap='viridis', alpha=0.5)

plt.show()


# ## Gaussian Mixture Model Algorithm

# In[25]:


# Perform hotspot detection using Gaussian Mixture Models
gmm = GaussianMixture(n_components=3)
gmm.fit(data)
labels = gmm.predict(data)


# In[26]:


normal_labels2 = labels.tolist()


# In[27]:


# Calculate the silhouette score
silhouette_score = silhouette(data, normal_labels2, metric='euclidean')

print("Silhouette Score:", silhouette_score)


# In[28]:


# Plot the data points with different colors for each cluster
m = Basemap(projection='merc',
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=20,
            resolution='c')

latitudes = data.loc[:, 'Latitude']
longitudes = data.loc[:, 'Longitude']
cases = data.loc[:, 'Confirmed']
x, y = m(longitudes, latitudes)
m.drawcoastlines()
m.drawcountries()
m.scatter(x, y, s=cases*0.005, c=labels, cmap='viridis', alpha=0.5)

plt.show()


# ## Hierarchical Clustering Algorithm

# In[29]:


# Perform hierarchical clustering
model = AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(data)


# In[30]:


normal_labels3 = labels.tolist()


# In[31]:


# Calculate the silhouette score
silhouette_score = silhouette(data, normal_labels3, metric='euclidean')

print("Silhouette Score:", silhouette_score)


# In[32]:


# Plot the data points with different colors for each cluster
m = Basemap(projection='merc',
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=20,
            resolution='c')

latitudes = data.loc[:, 'Latitude']
longitudes = data.loc[:, 'Longitude']
cases = data.loc[:, 'Confirmed']
x, y = m(longitudes, latitudes)
m.drawcoastlines()
m.drawcountries()
m.scatter(x, y, s=cases*0.005, c=labels, cmap='viridis', alpha=0.5)

plt.show()


# ## Mean-Shift Clustering Algorithm

# In[33]:


# Fit the Mean Shift clustering model
clustering = MeanShift().fit(data)
labels = clustering.labels_


# In[34]:


normal_labels4 = labels.tolist()


# In[35]:


# Calculate the silhouette score
silhouette_score = silhouette(data, normal_labels4, metric='euclidean')

print("Silhouette Score:", silhouette_score)


# In[36]:


# Plot the data points with different colors for each cluster
m = Basemap(projection='merc',
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=20,
            resolution='c')

latitudes = data.loc[:, 'Latitude']
longitudes = data.loc[:, 'Longitude']
cases = data.loc[:, 'Confirmed']
x, y = m(longitudes, latitudes)
m.drawcoastlines()
m.drawcountries()
m.scatter(x, y, s=cases*0.005, c=labels, cmap='viridis', alpha=0.5)

plt.show()


# # Accuracy

# ## Silhouette Score

# In[40]:


x = ['K-Means', 
     'Birch Algorithm', 
     'Gaussian Mixture Model', 
     'Hierarchical Clustering', 
     'Mean-Shift Clustering']
y = [0.8578287277073331, 
     0.9409221980378023, 
     0.5244693754087414, 
     0.9427003490280537, 
     0.8627866640325252]
#plt.scatter(x, y)
plt.title("Silhouette Score")
plt.plot(x, y, 'b*--')
plt.xticks(rotation = 30)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.show()


# ## Heatmap

# In[38]:


corr_matrix = df.corr()


# In[39]:


sns.heatmap(corr_matrix, cmap='coolwarm')

