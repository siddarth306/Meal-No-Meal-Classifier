from sklearn.svm import SVC

class SVM:
    def __init__(self):
        self.__svm = None

    def train(self, X, Y):
        self.__svm = SVC(kernel='poly', degree=8, gamma='scale')
        self.__svm.fit(X, Y)

    def test(self, X):
        self.__svm.predict(X)
