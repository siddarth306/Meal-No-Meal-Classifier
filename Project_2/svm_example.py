import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

# Preprocessing
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Ploynomial Kernel
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8, gamma='scale')
svclassifier.fit(X_train, y_train)

# Making Predictions
y_pred = svclassifier.predict(X_test)

# Evaulating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Another way of evaluating the algorithm using k-fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svclassifier, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))