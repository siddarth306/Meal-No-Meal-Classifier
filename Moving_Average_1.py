import numpy as np
from utilities import parse, parse_and_interpolate

# data = [range(1,42)]
filename = 'CGMSeriesLunchPat1.csv'
data = parse_and_interpolate(filename)

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
def compute_moving_avg(data):
    window_size, fixed_overlap = find_optimal_windowsize(data)
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

print(compute_moving_avg(data))

