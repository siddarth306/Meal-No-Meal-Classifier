import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# fake_Feat_matrix = [
#                     [1,2,3,4,5],
#                     [6,7,8,9,10],
#                     [11,12,13,14,15],
#                      [1,2,3,4,5],
#                     [6,7,8,9,10],
#                     [11,12,13,14,15],
#                     ]

def performPCA (FM):
    feat_matrix = np.array(FM)

    # Standardizing the features
    x = StandardScaler().fit_transform(feat_matrix)

    # compute pca
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
    , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principle component 4', 'principle component 5'])

    # explained variance tells you how much information (variance)
    #  can be attributed to each of the principal components.
    print ("\nExplained Variance", pca.explained_variance_ratio_)
    print(principalDf)
    return principalComponents
