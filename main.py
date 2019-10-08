from utilities import parse_and_interpolate
import numpy as np
from PCA_2 import performPCA

def main():
	files = ['CGMSeriesLunchPat1.csv', 'CGMSeriesLunchPat2.csv', 'CGMSeriesLunchPat3.csv', 'CGMSeriesLunchPat4.csv', 'CGMSeriesLunchPat5.csv'] 
	
	data = parse_and_interpolate(files[0])
	fft_features = get_fft_features(data)
	moving_avg = numpy.array(moving_avg(data))
	moving_kurt = moving_kurtosis(data)
	
	for index in range(1, len(files)):
		data = parse_and_interpolate(file[index])
	
		fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
		moving_avg = np.concatenate((moving_avg, numpy.array(moving_avg(data))), axis=0)
		moving_kurt = np.concatenate((moving_kurt, moving_kurtosis(data)), axis=0)


	feature_mattrix = np.concatenate((moving_avg, moving_kurt, fft_features), axis=1)
	pca_matrix = performPCA(feature_mattrix)

	plot(pca_matrix)
	plot(fft_features)
	plot(moving_avg)
	plot(moving_kurt)

main()