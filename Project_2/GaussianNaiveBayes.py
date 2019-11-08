from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes():
    def __init__(self):
        __gnb = None

    def train(X,Y):
        __gnb = GaussianNB()
        __gnb.fit(X, Y)

    def test(X):
        return __gnb.predict(X)
