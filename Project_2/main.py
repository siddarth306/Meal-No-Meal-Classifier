from utilities import *
import numpy as np
import matplotlib.pyplot as plt
import math
from feature_selection import *
import PCA as p
import svm as s
import ann as a
import DecisionTree as d
from sklearn.model_selection import KFold
import adaBoost as ab


# 4 classifiers used -> SVM, ANN, Ada Boost, Decision Trees

# plot graph
def plot(data, title, color):
	plt.scatter(range(len(data)), data, c=color)
	plt.title(title)
	plt.xlabel("Time Series Data")
	plt.ylabel("Feature Values")
	plt.grid(True)
	plt.savefig(title)
	plt.show()


# get Feature mattrix for new test data
def get_test_feature_matrix(filename, PCA):
	#import pdb; pdb.set_trace()
	data = parse_and_interpolate_test(filename)[0]
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = moving_avg(data)
	normal_skew_feature = normal_skew(data)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)

	#get PCA mattrix
	feature_mattrix = PCA.usePCA(feature_mattrix)
	return feature_mattrix


# get Feature mattrix for given data
def get_feature_mattrix():

	meal_files =	[
		   'MealNoMealData/mealData1.csv', 'MealNoMealData/mealData2.csv', 
		   'MealNoMealData/mealData3.csv', 'MealNoMealData/mealData4.csv', 
		   'MealNoMealData/mealData5.csv',
		]
	no_meal_files = [
		   'MealNoMealData/Nomeal1.csv','MealNoMealData/Nomeal2.csv',
		   'MealNoMealData/Nomeal3.csv','MealNoMealData/Nomeal4.csv',
		   'MealNoMealData/Nomeal5.csv'
		]

	   
	meal_data = parse_and_interpolate(meal_files)
	no_meal_data = parse_and_interpolate(no_meal_files)

	#------------------- for meal data-----------------------------
	#----------------------label = 1-------------------------------
	data = meal_data[0]
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = moving_avg(data)
	normal_skew_feature = normal_skew(data)

	for index in range(1, len(meal_data)):
		data = meal_data[index]
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg_features = np.concatenate((moving_avg_features, moving_avg(data)), axis=0)
		entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0) 
		normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

	feature_mattrix = np.concatenate((moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)
	np.set_printoptions(suppress=True)

	#get PCA mattrix2 and add label
	PCA = p.cal_PCA()
	pca_matrix1 = PCA.performPCA(feature_mattrix)
	temp = np.ones((pca_matrix1.shape[0], 1))
	pca_matrix1 = np.concatenate((pca_matrix1, temp), axis=1)


	#------------------- for no meal data-----------------------------
	#-----------------------label = 0-------------------------------
	data = no_meal_data[0]
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = moving_avg(data)
	normal_skew_feature = normal_skew(data)

	for index in range(1, len(no_meal_data)):
		data = no_meal_data[index]
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg_features = np.concatenate((moving_avg_features, moving_avg(data)), axis=0)
		entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0) 
		normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)

	#get PCA mattrix2 and add label
	pca_matrix2 = PCA.usePCA(feature_mattrix)
	temp = np.zeros((pca_matrix2.shape[0], 1))
	pca_matrix2 = np.concatenate((pca_matrix2, temp), axis=1)

	#combine mattrix
	feature_mattrix = np.concatenate((pca_matrix1, pca_matrix2), axis=0)

	#shuffle
	np.random.shuffle(feature_mattrix)

	return feature_mattrix, PCA


# Perfrom K-fold cross validation on all 4 models (k here is 4)
def kFold(data):
	kf = KFold(4,True,1)
	evals1 = []
	evals2 = []
	evals3 = []
	evals4 = []
	for train_index, test_index in kf.split(data):
		train_set , test_set = data[train_index], data[test_index]
		train_x, train_y = train_set[:, :-1], train_set[:, -1]
		test_x, test_y = test_set[:, :-1], test_set[:, -1]
		models = training(train_x, train_y)
		m1, m2, m3, m4 = testing(models, test_x, test_y)
		evals1.append(m1)
		evals2.append(m2)
		evals3.append(m3)
		evals4.append(m4)

	evals1 = np.array(evals1)
	evals2 = np.array(evals2)
	evals3 = np.array(evals3)
	evals4 = np.array(evals4)
	result_svm = np.mean(evals1, axis=0)
	result_ann = np.mean(evals2, axis=0)
	result_ada = np.mean(evals3, axis=0)
	result_dt = np.mean(evals4, axis=0)

	resultlabel = ["Accuracy: ","Precision: ", "Recall: ", "F1 score: "]

	print("K-Fold result")

	print("\nSVM results:")
	for idx, i in enumerate(list(result_svm)):
		print(resultlabel[idx],i)

	print("\nANN results:")
	for idx, i in enumerate(result_ann):
		print(resultlabel[idx],i)

	print("\nAda boost results:")
	for idx, i in enumerate(list(result_ada)):
		print(resultlabel[idx],i)

	print("\nDecision Tree results:")
	for idx, i in enumerate(result_dt):
		print(resultlabel[idx],i)
	return

# train all the 4 models
def training(x, y):
	models = []
	svm = s.SVM()
	svm.train(x,y)
	
	ann = a.Ann()
	ann.train(x,y)
	
	ada = ab.AdaBoost()
	ada.train(x,y)

	dt = d.DecisionTree()
	dt.train(x,y)

	models.append(svm)
	models.append(ann)
	models.append(ada)
	models.append(dt)
	return models


# test all the 4 models
def testing(models, x, y=None):
	evals = []

	svm = models[0]
	ann = models[1]
	ada = models[2]
	dt = models[3]

	evals.append(svm.test(x, y))
	evals.append(ann.test(x, y))
	evals.append(ada.test(x, y))
	evals.append(dt.test(x, y))
	return tuple(evals)


def main():
	
	# get feature mattrix for the given dataset
	feature_mattrix, PCA = get_feature_mattrix()

	# kfold to evaluate models
	kFold(feature_mattrix)

	# train on the given dataset
	train_x, train_y = feature_mattrix[:, :-1], feature_mattrix[:, -1]
	models = training(train_x, train_y)

	filename = "1"
	# input data file and get feature mattrix for new data
	while True:

		filename = input("Input filename (Enter 0 to exit): ")
		if filename == "0":
			break
		try:
			test_data = get_test_feature_matrix([filename], PCA)
		except Exception as e:
			print("Error Cannot open file")
			continue

		# predict new data using learned models
		result = testing(models, test_data)
		result = np.array(result)
		result_labels = ["\nSVM:\n","\nANN:\n", "\nAda Boost:\n", "\nDecision Tree:\n"]
		for i in range(len(result_labels)):
			print(result_labels[i], result[i])



main()	
