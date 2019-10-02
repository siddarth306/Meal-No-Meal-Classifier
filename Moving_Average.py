from utilities import parse
import numpy as np

filename = 'test.csv'
data = parse(filename)

def moving_avg(data):
    # settings
    window_size = 10
    overlap = 5
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