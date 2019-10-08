# This file prints out covariance matrix, eigenvalues, and transformed matrix using a data set at the URL below
# TODO: update 
#   -file to be a method that takes in our feature matrix 
#   -make sure the output matrix is what the assigment calls for
#
# Referenced to: https://plot.ly/ipython-notebooks/principal-component-analysis/

import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

# split data table into data X and class labels y
x = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# standardize
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x)

# find covariance matrix
import numpy as np
cov_mat = np.cov(X_std.T)
print('NumPy covariance matrix: \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# how much info can be attributed to each principle component
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print ('\nVariance Explained', var_exp)

# transform into 2D feature space
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('\nNew Matrix W:\n', matrix_w)
