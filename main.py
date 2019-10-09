from utilities import parse_and_interpolate
import numpy as np
from PCA_2 import performPCA
import matplotlib.pyplot as plt
import math
from feature_selection import *


def plot(data):
	plt.scatter(range(len(data)), data)
	#plt.ylim(-200, 500)
	plt.show()


def main():
	files = ['CGMSeriesLunchPat1.csv', 'CGMSeriesLunchPat2.csv', 'CGMSeriesLunchPat3.csv', 'CGMSeriesLunchPat4.csv', 'CGMSeriesLunchPat5.csv'] 
	
	data = parse_and_interpolate(files[0])
	fft_features = get_fft_features(data)
	moving_kurt_features = moving_kurt(data)
	#moving_avg_features = np.array(moving_avg(data))

	for index in range(1, len(files)):
		data = parse_and_interpolate(files[index])
		#print(data.shape)
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		#print(moving_avg_features.shape,"***************************")
		#print(np.array(moving_avg(data)).shape,"-----------------------")
		#moving_avg_features = np.concatenate((moving_avg_features, np.array(moving_avg(data))), axis=0)
		moving_kurt_features = np.concatenate((moving_kurt_features, moving_kurt(data)), axis=0)


	feature_mattrix = np.concatenate(( moving_kurt_features, fft_features), axis=1)
	print(feature_mattrix.shape)

	pca_matrix = performPCA(feature_mattrix)
	print(pca_matrix.shape)
	print(pca_matrix)

	plot(pca_matrix[:,4])
	# plot(fft_features)
	# plot(moving_avg)
	# plot(moving_kurt)

main()