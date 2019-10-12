from utilities import parse_and_interpolate
import numpy as np
from PCA_2 import performPCA
import matplotlib.pyplot as plt
import math
from feature_selection import *


def plot(data, title, color):
	plt.scatter(range(len(data)), data, c=color)
	plt.title(title)
	plt.xlabel("Time Series Data")
	plt.ylabel("Feature Values")
	plt.grid(True)
	#plt.ylim(y_min, y_max)
	plt.savefig(title)
	plt.show()


def main():
	files = ['CGMSeriesLunchPat1.csv', 'CGMSeriesLunchPat2.csv', 'CGMSeriesLunchPat3.csv', 'CGMSeriesLunchPat4.csv', 'CGMSeriesLunchPat5.csv'] 
	
	data = parse_and_interpolate(files[0])
	fft_features = get_fft_features(data)
	#moving_kurt_features = moving_kurt(data)
	entropy_feature = get_entropy(data)
	moving_avg_features = np.array(moving_avg(data))
	normal_skew_feature = normal_skew(data)

	for index in range(1, len(files)):
		data = parse_and_interpolate(files[index])
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg_features = np.concatenate((moving_avg_features, np.array(moving_avg(data))), axis=0)
		#moving_kurt_features = np.concatenate((moving_kurt_features, moving_kurt(data)), axis=0)
		entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0) 
		normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

	feature_mattrix = np.concatenate(( moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)
	np.set_printoptions(suppress=True)
	print(feature_mattrix)
	print(feature_mattrix.shape)

	pca_matrix = performPCA(feature_mattrix)

	# -----------------plot moving average features--------------------------
	# plot(moving_avg_features[:,0], "Moving Average 1", "blue")
	plot(moving_avg_features[:,1], "Moving Average 2", "blue")
	# plot(moving_avg_features[:,2], "Moving Average 3", "blue")
	# plot(moving_avg_features[:,3], "Moving Average 4", "blue")
	# plot(moving_avg_features[:,4], "Moving Average 5", "blue")

	# -----------------plot FFT features--------------------------
	# plot(fft_features[:,0], "FFT 1", "black")
	# plot(fft_features[:,1], "FFT 2", "black")
	plot(fft_features[:,2], "FFT 3", "black")
	# plot(fft_features[:,3], "FFT 4", "black")
	# plot(fft_features[:,4], "FFT 5", "black")

	# ----------------plot Entropy feature--------------------------
	plot(entropy_feature, "Entropy", "orange")

	# ----------------plot Skewness feature--------------------------
	plot(normal_skew_feature, "Skewness", "red")

	# ----------------------plot moving Kurtosis features-------------------------------
	# ----------------This was not the part of final four types of features-------------
	# plot(moving_kurt_features[:,0], "Moving Kurtosis 1", "orange")
	# plot(moving_kurt_features[:,1], "Moving Kurtosis 2", "orange")
	# plot(moving_kurt_features[:,2], "Moving Kurtosis 3", "orange")
	# plot(moving_kurt_features[:,3], "Moving Kurtosis 4", "orange")
	# plot(moving_kurt_features[:,4], "Moving Kurtosis 5", "orange")
	# plot(moving_kurt_features[:,5], "Moving Kurtosis 6", "orange")
	# plot(moving_kurt_features[:,6], "Moving Kurtosis 7", "orange")
	# plot(moving_kurt_features[:,7], "Moving Kurtosis 8", "orange")
	# plot(moving_kurt_features[:,8], "Moving Kurtosis 9", "orange")

	# -----------------plot moving Skewness features------------------------------------
	# ----------------This was not the part of final four types of features-------------
	# plot(moving_skew_feature[:,0], "Moving Skewness 1", "red")
	# plot(moving_skew_feature[:,1], "Moving Skewness 2", "red")
	# plot(moving_skew_feature[:,2], "Moving Skewness 3", "red")
	# plot(moving_skew_feature[:,3], "Moving Skewness 4", "red")
	# plot(moving_skew_feature[:,4], "Moving Skewness 5", "red")
	# plot(moving_skew_feature[:,5], "Moving Skewness 6", "red")
	# plot(moving_skew_feature[:,6], "Moving Skewness 7", "red")
	# plot(moving_skew_feature[:,7], "Moving Skewness 8", "red")
	# plot(moving_skew_feature[:,8], "Moving Skewness 9", "red")

	# -----------------plot PCA--------------------------
	# plot(pca_matrix[:,0], "Principal Component 1", "green")
	# plot(pca_matrix[:,1], "Principal Component 2", "green")
	# plot(pca_matrix[:,2], "Principal Component 3", "green")
	# plot(pca_matrix[:,3], "Principal Component 4", "green")
	# plot(pca_matrix[:,4], "Principal Component 5", "green")

main()
