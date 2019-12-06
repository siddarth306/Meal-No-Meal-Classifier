import numpy as np
import math
import pandas as pd
from sklearn.impute import SimpleImputer
import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
from impyute.imputation.cs import mice




def parse_and_interpolate_test(filename_list):

    all_dfs = []
    len_list = [0]*len(filename_list)
   
    for idx,i in enumerate(filename_list):
        df = pd.read_csv(i)
        df_lists = df.values.tolist()
   

        all_dfs = all_dfs + df_lists
        len_list[idx] = len(df_lists)
    filled_df_list = []
    if np.isnan(np.array(all_dfs)).any():
        filled_df = mice(np.array(all_dfs))
    else:
        filled_df = np.array(all_dfs)

    result = []
    start = 0
    for i in len_list:
        result.append(filled_df[start:start+i])
        start = i

    return result




def parse_and_interpolate(filename_list):

    all_dfs = []
    len_list = [0]*len(filename_list)
   
    for idx,i in enumerate(filename_list):
        df = pd.read_csv(i)
        df_lists = df.values.tolist()
   

        valid_df_list = []
        for list_idx,each_list in enumerate(df_lists):
            nanPercent = sum([1  if math.isnan(val) else 0 for val in each_list]) / len(each_list)
            #nan_check = np.isnan(np.array(each_list)).all()
            if nanPercent < 0.3:
                valid_df_list.append(each_list)
        all_dfs = all_dfs + valid_df_list
        len_list[idx] = len(valid_df_list)
    filled_df_list = []
    if np.isnan(np.array(all_dfs)).any():
        filled_df = mice(np.array(all_dfs))
    else:
        filled_df = np.array(all_dfs)

    result = []
    start = 0
    for i in len_list:
        result.append(filled_df[start:start+i])
        start = i

    return result


#def parse_and_interpolate(filename):
#    df = pd.read_csv(filename)
#    df_lists = df.values.tolist()
#    filled_df_list = []
#    #import pdb;pdb.set_trace()
#    for list_idx,each_list in enumerate(df_lists):
#        #nanPercent = sum([1  if math.isnan(val) else 0 for val in each_list]) / len(each_list)
#        if not all([math.isnan(val) for val in each_list]):
#        #if nanPercent < 0.8:
#            #cleaned_data = pd.Series(each_list).interpolate(method='linear').tolist()
#            #imp_mean = SimpleImputer( strategy='most_frequent')
#            #imp_mean.fit(each_list)
#            #cleaned_data = imp_mean.transform(each_list)
#            #if not np.any(np.isnan(cleaned_data)) and np.all(np.isfinite(cleaned_data)):
#            filled_df_list.append(each_list)
#   
#    #imp_mean = SimpleImputer( strategy='most_frequent')
#    #imp_mean.fit(filled_df_list)
#    filled_df = mice(np.array(filled_df_list))
#    #filled_df = fast_knn(np.array(df_lists), k=30)
#    #cleaned_data = imp_mean.transform(filled_df_list)
#    #filled_df = pd.DataFrame(cleaned_data, columns=df.columns)
#
#    return np.array(filled_df)


def parse(filename):
    r = np.genfromtxt(filename, delimiter=',', names=True, case_sensitive=True, dtype=np.float64)
    return r


