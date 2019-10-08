from utilities import parse, parse_and_interpolate
import numpy as np
import math
from PCA_2 import performPCA

filename = 'CGMSeriesLunchPat1.csv'
data = parse_and_interpolate(filename)
data = [range(100)]
print(data)

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
                print(col)

            print ('\n')
            col = col-overlap
            moving_avg.append(tmp_sum)

        print(len(moving_avg))   
        result.append(moving_avg)
    return result

mvg_avg = moving_avg(data)
# Assuming 
# performPCA(mvg_avg)