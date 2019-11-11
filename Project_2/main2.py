from utilities2 import parse_and_interpolate
import numpy as np
#from PCA_2 import performPCA
import matplotlib.pyplot as plt
import math
from feature_selection2 import *
import PCA as p
import svm as s
import ann as a
import GaussianNaiveBayes as g
import DecisionTree as d
from sklearn.model_selection import KFold


def plot(data, title, color):
	plt.scatter(range(len(data)), data, c=color)
	plt.title(title)
	plt.xlabel("Time Series Data")
	plt.ylabel("Feature Values")
	plt.grid(True)
	#plt.ylim(y_min, y_max)
	plt.savefig(title)
	plt.show()


def get_test_feature_matrix(filename, PCA):
	#import pdb; pdb.set_trace()
	data = parse_and_interpolate(filename)[0]
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = moving_avg(data)
	normal_skew_feature = normal_skew(data)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)

	#get PCA mattrix2 and add label
	feature_mattrix = PCA.usePCA(feature_mattrix)
	return feature_mattrix



def get_feature_mattrix():
	


	meal_files = [
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
	result_gnb = np.mean(evals3, axis=0)
	result_dt = np.mean(evals4, axis=0)

	resultlabel = ["Accuracy: ","Precision: ", "Recall: ", "F1 score: "]

	print("K-Fold result")

	print("SVM results:")
	for idx, i in enumerate(list(result_svm)):
		print(resultlabel[idx],i)

	print("ANN results:")
	for idx, i in enumerate(result_ann):
		print(resultlabel[idx],i)

	print("Gaussian Naive Bayes results:")
	for idx, i in enumerate(list(result_gnb)):
		print(resultlabel[idx],i)

	print("Decision Tree results:")
	for idx, i in enumerate(result_dt):
		print(resultlabel[idx],i)
	return


def training(x, y):
	models = []
	svm = s.SVM()
	svm.train(x,y)
	
	ann = a.Ann()
	ann.train(x,y)
	
	gnb = g.GaussianNaiveBayes()
	gnb.train(x,y)

	dt = d.DecisionTree()
	dt.train(x,y)

	models.append(svm)
	models.append(ann)
	models.append(gnb)
	models.append(dt)
	return models

	# split the data from the class label
	#x, y = data[:, :-1], data[:, -1]

	#if algo == "SVM":
	#	SVM = s.SVM()
	#	SVM.train(x,y)
	#	SVM.k_fold_validate(x,y) # perform k-fold cross validation


def testing(models, x, y=None):
	evals = []

	svm = models[0]
	ann = models[1]
	gnb = models[2]
	dt = models[3]

	evals.append(svm.test(x, y))
	evals.append(ann.test(x, y))
	evals.append(gnb.test(x, y))
	evals.append(dt.test(x, y))
	return tuple(evals)




def main():
	
	feature_mattrix, PCA = get_feature_mattrix() # need the PCA object to fit testing

	#kfold to evaluate model
	kFold(feature_mattrix)

	train_x, train_y = feature_mattrix[:, :-1], feature_mattrix[:, -1]
	models = training(train_x, train_y)

	filename = input("Input filename: ")
	test_data = get_test_feature_matrix([filename], PCA) # need the PCA object to fit testing

	print(testing(models, test_data))

	#test
	# train machines
	# training( feature_mattrix)
	# training("ANN", feature_mattrix)




main()	
