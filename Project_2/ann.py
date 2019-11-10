import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import *


class Ann:
	def __init__(self):
		self.model = None

	def train(self, X_train, Y_train):

		self.model=keras.Sequential([
			keras.layers.Dense(20,activation=tf.nn.relu),
			keras.layers.Dense(15,activation=tf.nn.relu),
			keras.layers.Dense(2,activation=tf.nn.softmax)
			])

		self.model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
		self.model.fit(X_train, Y_train, batch_size= 20, epochs=100, verbose=0)

	def test(self, X, Y):

		y_pred = self.model.predict(X)
		y_pred = np.argmax(y_pred, axis=1)

		if Y is not None:
			report = classification_report(Y, y_pred, output_dict=True)['1.0']
			accuracy = accuracy_score(Y, y_pred, normalize=True)
			#test_loss, test_acc = self.model.evaluate(X_test, Y_test)
			return [accuracy, report['precision'], report['recall'], report['f1-score']]
		return y_pred
		
		# if Y_test is not None:
		# 	test_loss, test_acc = self.model.evaluate(X_test, Y_test)
		# 	return [test_acc]
		# pred = self.model.predict(X_test)
		# return np.argmax(pred, axis=1)