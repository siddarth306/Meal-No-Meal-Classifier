from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    def __init__(self):
        self.__gnb = None

    def train(self, X, Y):
        self.__gnb = GaussianNB()
        self.__gnb.fit(X, Y)

    def test(self, X):
        return self.__gnb.predict(X)
