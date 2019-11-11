from sklearn import tree
from sklearn.metrics import *

class DecisionTree:
	def __init__(self):
		self.__dt = None

	def train(self, X,Y):
		self.__dt = tree.DecisionTreeClassifier()
		self.__dt.fit(X, Y)

	def test(self, X, Y):
		y_pred = self.__dt.predict(X)
		if Y is not None:
			report = classification_report(Y, y_pred, output_dict=True)['1.0']
			accuracy = accuracy_score(Y, y_pred, normalize=True)
			return [accuracy, report['precision'], report['recall'], report['f1-score']]
		return y_pred
