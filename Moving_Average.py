from utilities import parse, parse_and_interpolate
import numpy as np

filename = 'CGMSeriesLunchPat1.csv'
data = parse_and_interpolate(filename)

def moving_avg(data):
    # settings
    window_size = len(data[0]) // 5
    overlap = window_size // 2
    result = []

    for row_data in data:
        col = 0
        moving_avg = []
        while col < len(row_data)-5:
            tmp_sum = 0

            for i in range(window_size):
                if col == len(row_data):
                    break
                tmp_sum = tmp_sum + row_data[col]
                col = col + 1

            col = col-overlap
            moving_avg.append(tmp_sum)

        result.append(moving_avg)
    return result

print(moving_avg(data))
