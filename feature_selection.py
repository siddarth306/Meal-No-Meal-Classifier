from utilities import parse, parse_and_interpolate
from scipy.stats import kurtosis,skew
import numpy as np
import pandas as pd 
import tsfresh
import math
from PCA_2 import performPCA

def count_window_size(data, fixed_overlap, window_size):
    window_ct = 0
    ix = 0
    exit = 0
    #count the windows
    while ix < len(data):
        for i in range(window_size):
            if(ix==len(data)):
                exit = 1
                break
            # print(data[ix])
            ix=ix+1
        
        window_ct+=1
        if(exit==1):
            break
        # print('\n')
        ix=ix-fixed_overlap

    # print("window_size: ", window_ct)
    return window_ct

def find_optimal_windowsize(data):
    # start setting
    fixed_overlap = int(len(data)/5/2) 
    window_size = len(data)//5 

    # find the window size that will give 5 window output
    ct = count_window_size(data, fixed_overlap, window_size)
    while ct!=5:
        window_size=window_size+1
        ct = count_window_size(data, fixed_overlap, window_size)

    print ("Optimal window_size:", window_size, " for overlap:", fixed_overlap)
    return window_size, fixed_overlap


#bug will fail if data len=11 because it will fail to find windowsize=5
#it will work for our case where data len=31 or len=42
def moving_avg(data):
    window_size, fixed_overlap = find_optimal_windowsize(data[0])
    result = []

    for row_data in data:
        ix = 0
        exit=0
        moving_avg = []
        
        while ix < len(row_data):
            tmp_sum=0
            for i in range(window_size):
                if(ix==len(row_data)):
                    exit = 1
                    break
                # print(row_data[ix])
                tmp_sum+=row_data[ix]
                ix=ix+1

            moving_avg.append(tmp_sum/window_size)
            ix=ix-fixed_overlap
            if(exit==1):
                break
            # print('\n')
        
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

def moving_skew(data):
    window_size = math.ceil(len(data[0]) / 12.0)
    #overlap = math.ceil(window_size / 2.0)
    overlap = 2
    result = []

    for row_data in data:
        col = 0
        skew_list = []
        while col < len(row_data)-window_size:

            skewness = skew(row_data[col:col+window_size + 1])
            skew_list.append(skewness)
            col += window_size - overlap

        result.append(skew_list)
    return result


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


