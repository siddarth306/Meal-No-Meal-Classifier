import tensorflow as tf
from tensorflow import keras
import numpy as np


class Ann:
	def __init__(self):
		self.model = None

	def train(self, data):

		X_train = data[:,:5]
		Y_train = data[:, 5]
		self.model=keras.Sequential([
			keras.layers.Dense(10,activation=tf.nn.sigmoid),
			keras.layers.Dense(5,activation=tf.nn.sigmoid),
			keras.layers.Dense(2,activation=tf.nn.softmax)
			])

		self.model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
		self.model.fit(X_train, Y_train, batch_size= 1500, epochs=1000)

	def test(self, data):
		X_test = data[:,:5]
		Y_test = data[:, 5]
		test_loss, test_acc = self.model.evaluate(X_test, Y_test)