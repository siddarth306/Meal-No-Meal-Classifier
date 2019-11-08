import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self):
        self.pca = None

    def performPCA(self, FM):
        feat_matrix = np.array(FM)

        # Standardizing the features
        x = StandardScaler().fit_transform(feat_matrix)

        # compute pca
        self.pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
        , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principle component 4', 'principle component 5'])

        # explained variance tells you how much information (variance)
        #  can be attributed to each of the principal components.
        print ("\nExplained Variance", pca.explained_variance_ratio_)
        print(principalDf)
        return principalComponents


    def usePCA(self, FM):
        feat_matrix = np.array(FM)

        x = StandardScaler().fit_transform(feat_matrix)

        principalComponents = pca.transform(x)
        principalDf = pd.DataFrame(data = principalComponents
        , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principle component 4', 'principle component 5'])

        # explained variance tells you how much information (variance)
        #  can be attributed to each of the principal components.
        print ("\nExplained Variance", pca.explained_variance_ratio_)
        print(principalDf)
        return principalComponents
