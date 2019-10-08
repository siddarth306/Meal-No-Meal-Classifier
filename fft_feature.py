import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import tsfresh


def get_data():
	data1_1 = np.array(pd.read_csv('CGMSeriesLunchPat1.csv'))
	data1_2 = np.array(pd.read_csv('CGMSeriesLunchPat2.csv'))
	data1_3 = np.array(pd.read_csv('CGMSeriesLunchPat3.csv'))
	data1_4 = np.array(pd.read_csv('CGMSeriesLunchPat4.csv'))
	data1_5 = np.array(pd.read_csv('CGMSeriesLunchPat5.csv'))

	data1 = np.concatenate((data1_1, data1_2), axis=0)
	data1 = np.concatenate((data1, data1_3), axis=0)
	data1 = np.concatenate((data1, data1_4[:,:31]), axis=0)
	data1 = np.concatenate((data1, data1_5), axis=0)

	return data1

def get_fft_features(data1):
	feature1 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 10, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature2 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 11, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature3 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 12, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature4 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 13, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature5 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 14, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	featute_matrix = np.array([feature1, feature2, feature3, feature4, feature5])
	return featute_matrix

def main():
	data = get_data()
	featute_matrix = get_fft_features(data)
	print(featute_matrix.shape)


main()


