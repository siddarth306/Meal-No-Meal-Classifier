from sklearn.svm import SVC

class SVM():
    def __init__(self):
        __svm = None

    def train(X,Y):
        __svm = SVC(kernel='poly', degree=8, gamma='scale')
        __svm.fit(X, Y)

    def test(X):
        return __svm.predict(X)
