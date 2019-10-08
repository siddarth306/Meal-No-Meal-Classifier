from scipy.stats import kurtosis
from utilities import parse_and_interpolate
import math


def moving_kurt(data):
    # settings
    window_size = math.ceil(len(data[0]) / 5.0)
    overlap = math.ceil(window_size / 2.0)
    result = []

    for row_data in data:
        col = 0
        kurt_list = []
        while col < len(row_data)-window_size:

            print(row_data[col:col+window_size])
            kurt = kurtosis(row_data[col:col+window_size])
            col += window_size

            col = col-overlap
            kurt_list.append(kurt)

        result.append(kurt_list)
    return result

