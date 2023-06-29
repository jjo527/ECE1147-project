# ############################################################
# This code is from following the linked example below
# - https://youtu.be/2eDFjw456AM
# - https://github.com/siddiquiamir/Python-Clustering-Tutorials/blob/main/DBSCAN.ipynb
# ############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN


with open("DBSCAN_output.txt",'w') as file:

    df = pd.read_csv("Mall_customers.csv")

    head = df.head()
    file.write(f"{head}")
    file.write("\n...\n")
    tail = df.tail()
    file.write(f"{tail}")

    file.write("\n\n-----------------------------------\n")
    shape = df.shape
    file.write(f"{shape}\n\n")

    df = df.iloc[:, [3,4]].values
    file.write(f"{df}\n\n")

    plt.scatter(df[:,0], df[:,1], s=10, c= 'black')
    # plt.show()
    plt.savefig("fig1.png")

    file.write("-----------------------------------\n")

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
    # plt.show()
    plt.savefig("fig2.png")

    # -------------------------------------------
    dbscan = DBSCAN(eps=5, min_samples=5)
    labels = dbscan.fit_predict(df)
    unique = np.unique(labels)
    file.write(f"{unique}")

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