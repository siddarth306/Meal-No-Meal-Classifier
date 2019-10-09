from utilities import parse, parse_and_interpolate
from scipy.stats import kurtosis
import numpy as np
import pandas as pd 
import tsfresh
import math
from PCA_2 import performPCA



# moving_avg will always return 8 average values per row
def moving_avg(data):
    # settings
    window_size = math.ceil(len(data[0]) / 5.0)
    #overlap = math.ceil(window_size / 2.0)
    overlap = 0
    result = []

    for row_data in data:
        col = 0
        moving_avg = []
        while col < len(row_data)-window_size:
            tmp_sum = 0

            for i in range(window_size):
                if col == len(row_data):
                    break
                tmp_sum = tmp_sum + row_data[col]
                col = col + 1

            print ('\n')
            col = col-overlap
            moving_avg.append(tmp_sum)

        result.append(moving_avg)
    return result

def get_fft_features(data1):
	feature1 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 1, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature2 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 2, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature3 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 3, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature4 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 4, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	feature5 = [ abs(list(tsfresh.feature_extraction.feature_calculators.fft_coefficient(data1[i,:30], [{"coeff": 5, "attr": "real"}]))[0][1]) for i in range(len(data1)) ]
	featute_matrix = np.array([feature1, feature2, feature3, feature4, feature5])
	return featute_matrix.T


def moving_kurt(data):
    window_size = math.ceil(len(data[0]) / 7.0)
    #overlap = math.ceil(window_size / 2.0)
    overlap = 2
    result = []

    for row_data in data:
        col = 0
        kurt_list = []
        while col < len(row_data)-window_size:

            kurt = kurtosis(row_data[col:col+window_size + 1])
            kurt_list.append(kurt)
            col += window_size - overlap

        result.append(kurt_list)
    return result


