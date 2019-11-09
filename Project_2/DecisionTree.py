from sklearn import tree

class DecisionTree:
	def __init__(self):
		self.__dt = None

	def train(self, X,Y):
		self.__dt = tree.DecisionTreeClassifier()
		self.__dt.fit(X, Y)
	def test(self, X, Y):
		if Y is not None:
			return [self.__dt.score(X,Y)]
		return self.__dt.predict(X)