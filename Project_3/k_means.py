from sklearn.cluster import KMeans

class K_means:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = None

    def train(self, feature_mattrix):

        # K-Means (run k-means on data DataFrame formatted as a matrix
        self.model = KMeans(self.n_clusters, random_state=0).fit(feature_mattrix)

        # print(self.model.labels_)
        # print(self.model.cluster_centers_)

        return self.model.labels_, self.model.cluster_centers_


    def test(self, feature_mattrix):
        return self.model.predict(feature_mattrix) #labels
