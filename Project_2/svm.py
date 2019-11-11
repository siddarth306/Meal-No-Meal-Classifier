from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *

class SVM:
	def __init__(self):
		self.__svm = None

	def train(self, X, Y):
		self.__svm = SVC(kernel='rbf', gamma='scale')
		self.__svm.fit(X, Y)

	def test(self, X, Y=None):
		y_pred = self.__svm.predict(X)
		if Y is not None:
			report = classification_report(Y, y_pred, output_dict=True)['1.0']
			accuracy = accuracy_score(Y, y_pred, normalize=True)
			return [accuracy, report['precision'], report['recall'], report['f1-score']]
		return y_pred



