from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class SVM:
    def __init__(self):
        self.__svm = None

    def train(self, X, Y):
        self.__svm = SVC(kernel='poly', degree=8, gamma='scale')
        self.__svm.fit(X, Y)

    def test(self, X):
        return self.__svm.predict(X)

    def k_fold_validate(self, x, y):
        scores = cross_val_score(self.__svm, x, y, cv=4, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

