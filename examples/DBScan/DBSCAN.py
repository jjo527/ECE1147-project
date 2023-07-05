# %% [markdown]
# # DBSCAN

# %%
# ############################################################
# This code is from following the linked example below
# - https://youtu.be/2eDFjw456AM
# - https://github.com/siddiquiamir/Python-Clustering-Tutorials/blob/main/DBSCAN.ipynb
# ############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Mall_customers.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df = df.iloc[:, [3,4]].values

# %%
df

# %%
plt.scatter(df[:,0], df[:,1], s=10, c= "black")

# %%
from sklearn.cluster import KMeans

# %%
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i,
    init = 'k-means++', max_iter= 300, n_init= 10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# %%
from sklearn.cluster import DBSCAN

# %%
dbscan = DBSCAN(eps=5, min_samples=5)

# %%
labels = dbscan.fit_predict(df)

# %%
np.unique(labels)

# %%
# Visualising the clusters
plt.scatter(df[labels == -1, 0], df[labels == -1, 1], s = 10, c = 'black')

plt.scatter(df[labels == 0, 0], df[labels == 0, 1], s = 10, c = 'blue')
plt.scatter(df[labels == 1, 0], df[labels == 1, 1], s = 10, c = 'red')
plt.scatter(df[labels == 2, 0], df[labels == 2, 1], s = 10, c = 'green')
plt.scatter(df[labels == 3, 0], df[labels == 3, 1], s = 10, c = 'brown')
plt.scatter(df[labels == 4, 0], df[labels == 4, 1], s = 10, c = 'pink')
plt.scatter(df[labels == 5, 0], df[labels == 5, 1], s = 10, c = 'yellow')
plt.scatter(df[labels == 6, 0], df[labels == 6, 1], s = 10, c = 'silver')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


