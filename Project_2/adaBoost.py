from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *

class AdaBoost:
	def __init__(self):
		self.ab = None

	def train(self, X,Y):
		self.ab = AdaBoostClassifier(n_estimators=100, random_state=0)
		self.ab.fit(X, Y)

	def test(self, X, Y):
		y_pred = self.ab.predict(X)
		if Y is not None:
			report = classification_report(Y, y_pred, output_dict=True)['1.0']
			accuracy = accuracy_score(Y, y_pred, normalize=True)
			return [accuracy, report['precision'], report['recall'], report['f1-score']]
		return y_pred
