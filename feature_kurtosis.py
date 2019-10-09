from scipy.stats import kurtosis
from utilities import parse_and_interpolate
import math


def moving_kurt(data):
    offset = 0
    if len(data[0]) % 5 == 0:
        offset = 1
    window_size = math.ceil(len(data[0]) / 7.0)
    #overlap = 2
    overlap =2
    result = []

    for row_data in data:
        col = 0
        kurt_list = []
        while col < len(row_data)-window_size:

            kurt = kurtosis(row_data[col:col+window_size+1])
            kurt_list.append(kurt)
            #print(col, col+window_size-overlap)
            col += window_size - overlap
        result.append(kurt_list)
    return result

#for i in range(20,150):
#    data = [range(35)]
#    x = moving_kurt(data)
#    print(len(x[0]))
