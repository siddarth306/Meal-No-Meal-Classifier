import numpy as np
from scipy.spatial import distance

# # TEST DATA:
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
# data = np.array([[1, 2], [2, 2], [2, 3],
#              [8, 7], [8, 8], [25, 80]])
# print("original data: \n", data)
#
# db = DBSCAN(eps=3, min_samples=2).fit(data)
# print("\ndbscan labels: ", db.labels_)
# print("core points:{}".format(db.components_)) # not used
# km = KMeans(2).fit(data)
# print("\nk-means centriods: \n", km.cluster_centers_)
# print("k-means labels", km.labels_)

def SSE_dbscan(data, labels):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # group data points by cluster
    cluster_data = [] # init
    for i in range(0, n_clusters_):
        cluster_data.append([])

    for i in range(0, len(data)):
        # print (X[i], labels[i])
        if (labels[i]!= -1):
            cluster_data[labels[i]].append(list(data[i]))

    # compute the mean/centriods from each cluster
    centriods=[]
    for i in range (0, n_clusters_):
        mean=np.mean(cluster_data[i], axis=0)
        centriods.append(mean)

    # compute SSE for each cluster
    SSE_clusters = []
    for i in range(0, n_clusters_):
        sum_dist=0;
        for j in range(0, len(cluster_data[i])):
            dist = (distance.euclidean(cluster_data[i][j], centriods[i]))
            sum_dist = sum_dist + dist
        SSE_clusters.append(sum_dist)

    # Total SSE
    total_SSE = np.sum(SSE_clusters)

    # #print
    # for i in range(0, n_clusters_):
    #     print('cluster {}: {}'.format(i, cluster_data[i]))
    #     print('computed centriod:', centriods[i])
    #     print('SSE:', SSE_clusters[i])
    #     print("\n")
    # print('Total SSE: ', total_SSE)

    return total_SSE

def SSE_kmeans(data, labels, centriods):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # group data points by cluster
    cluster_data = [] # init
    for i in range(0, n_clusters_):
        cluster_data.append([])

    for i in range(0, len(data)):
        # print (X[i], labels[i])
        if (labels[i]!= -1):
            cluster_data[labels[i]].append(list(data[i]))

    # compute SSE for each cluster
    SSE_clusters = []
    for i in range(0, n_clusters_):
        sum_dist=0;
        for j in range(0, len(cluster_data[i])):
            dist = (distance.euclidean(cluster_data[i][j], centriods[i]))
            sum_dist = sum_dist + dist
        SSE_clusters.append(sum_dist)

    # Total SSE
    total_SSE = np.sum(SSE_clusters)

    # #print
    # for i in range(0, n_clusters_):
    #     print('cluster {}: {}'.format(i, cluster_data[i]))
    #     print('computed centriod:', centriods[i])
    #     print('SSE:', SSE_clusters[i])
    #     print("\n")
    # print('Total SSE: ', total_SSE)

    return total_SSE
