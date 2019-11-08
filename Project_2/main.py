from utilities import parse_and_interpolate
import numpy as np
#from PCA_2 import performPCA
import matplotlib.pyplot as plt
import math
from feature_selection import *
import PCA as p


def plot(data, title, color):
	plt.scatter(range(len(data)), data, c=color)
	plt.title(title)
	plt.xlabel("Time Series Data")
	plt.ylabel("Feature Values")
	plt.grid(True)
	#plt.ylim(y_min, y_max)
	plt.savefig(title)
	plt.show()


def get_feature_mattrix():
	files = [
			'MealNoMealData/mealData1.csv', 'MealNoMealData/mealData2.csv', 
			'MealNoMealData/mealData3.csv', 'MealNoMealData/mealData4.csv', 
			'MealNoMealData/mealData5.csv', 'MealNoMealData/Nomeal1.csv',
			'MealNoMealData/Nomeal2.csv', 'MealNoMealData/Nomeal3.csv',
			'MealNoMealData/Nomeal4.csv', 'MealNoMealData/Nomeal5.csv'
			]


	#------------------- for meal data-----------------------------
	#----------------------label = 1-------------------------------
	data = parse_and_interpolate(files[0])
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = np.array(moving_avg(data))
	normal_skew_feature = normal_skew(data)

	for index in range(1, int(len(files)/2)):
		data = parse_and_interpolate(files[index])
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg_features = np.concatenate((moving_avg_features, np.array(moving_avg(data))), axis=0)
		entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0) 
		normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)
	np.set_printoptions(suppress=True)

	PCA = p.cal_PCA()
	pca_matrix1 = PCA.performPCA(feature_mattrix)
	temp = np.ones((pca_matrix1.shape[0], 1))
	pca_matrix1 = np.concatenate((pca_matrix1, temp), axis=1)


	#------------------- for no meal data-----------------------------
	#-----------------------label = 0-------------------------------
	data = parse_and_interpolate(files[int(len(files)/2)])
	fft_features = get_fft_features(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = np.array(moving_avg(data))
	normal_skew_feature = normal_skew(data)

	for index in range(int(len(files)/2)+1, len(files)):
		data = parse_and_interpolate(files[index])
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg_features = np.concatenate((moving_avg_features, np.array(moving_avg(data))), axis=0)
		entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0) 
		normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)

	#get PCA mattrix2 and add label
	pca_matrix2 = PCA.usePCA(feature_mattrix)
	temp = np.zeros((pca_matrix2.shape[0], 1))
	pca_matrix2 = np.concatenate((pca_matrix2, temp), axis=1)

	#combine mattrix

	#shuffle


	#return combined mattrix

def training(algo, data):
	if algo == ANN:
		use_ann()



def main():
	feature_mattrix = get_feature_mattrix()
	training("ANN", feature_mattrix)




main()	
