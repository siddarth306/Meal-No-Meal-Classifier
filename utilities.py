import numpy as np
import math
import pandas as pd
from scipy.stats import kurtosis


def parse_and_interpolate(filename):
    df = pd.read_csv(filename)
    df_lists = df.values.tolist()
    filled_df_list = []
    for list_idx,each_list in enumerate(df_lists):
        cleaned_data = pd.Series(each_list).interpolate(method='linear').tolist()
        if not any([math.isnan(val) for val in cleaned_data]):
            filled_df_list.append(cleaned_data)
   
    filled_df = pd.DataFrame(filled_df_list, columns=df.columns)

    return np.array(filled_df)

def parse(filename):
    r = np.genfromtxt(filename, delimiter=',', names=True, case_sensitive=True, dtype=np.float64)
    return r


