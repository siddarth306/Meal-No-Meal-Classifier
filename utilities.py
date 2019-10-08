import numpy as np
import math
import pandas as pd

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


#def poly_coeff(data_matrix, timestamps):
#    import pdb; pdb.set_trace()
#    for data, timestamps in zip(data_matrix, timestamps):
#        data_list = data.tolist()
#        timestamps_list = timestamps.tolist()
#
#        if math.isnan(data.tolist()[-1]):
#            data_list = data_list[:-1]
#            timestamps_list = timestamps[:-1]
#        print(data)
#        data_int = [int(i) for i in data_list]
#        timestamps_int = [int(i) for i in timestamps_list]
#        p = np.polyfit(data_int, timestamps_int, deg=3, full=True)
#        coefs = np.poly1d(p)
#        print(len(coefs.r))
#        print(coefs.coef)
#        print(coefs.coeff)
#        print(coefs.coeffs)
#        print("d")
import pdb; pdb.set_trace()
data = parse_and_interpolate("data/CGMSeriesLunchPat3.csv")
timestamps = parse("data/CGMDatenumLunchPat3.csv")
#poly_coeff(data, timestamps)
