from utilities import parse, parse_and_interpolate
import numpy as np
import math
from PCA_2 import performPCA
import matplotlib.pyplot as plt

# # clean data (should be done in main)
# filename = 'CGMSeriesLunchPat1.csv'
# data = parse_and_interpolate(filename)


# moving_avg will always return 8 average values per row
def moving_avg(data):
    # settings
    window_size = math.ceil(len(data[0]) / 5.0)
    overlap = math.ceil(window_size / 2.0)
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

            col = col-overlap
            moving_avg.append(tmp_sum/window_size)

        result.append(moving_avg)
    return result


# #test
# mvg_avg = moving_avg(data)
# principleDf = performPCA(mvg_avg)

# #plot Principle component 1
# data = principleDf
# plt.scatter(range(len(data)), data[:,0])
# plt.show()