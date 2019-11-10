from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

class GaussianNaiveBayes:
	def __init__(self):
		self.__gnb = None

	def train(self, X, Y):
		self.__gnb = GaussianNB()
		self.__gnb.fit(X, Y)

	def test(self, X, Y=None):
		y_pred = self.__gnb.predict(X)
		if Y is not None:
			report = classification_report(Y, y_pred, output_dict=True)['1.0']
			accuracy = accuracy_score(Y, y_pred, normalize=True)
			return [accuracy, report['precision'], report['recall'], report['f1-score']]
		return y_pred
