from sklearn.cluster import DBSCAN

class Dbscan:
    def __init__(self, eps, min_samples):
        self.model = None
        self.ep = eps
        self.min_samp = min_samples

    def train(self, feature_matrix):
        self.model = DBSCAN(eps=self.ep, min_samples=self.min_samp).fit(feature_matrix)
        return self.model.labels_


    def test(self, feature_mattrix):
        return self.model.fit_predict(feature_mattrix) #labels
