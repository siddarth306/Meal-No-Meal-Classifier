from utilities import *
import numpy as np
import matplotlib.pyplot as plt
import math
from feature_selection import *
import k_means as km
import dbscan as db
from sklearn import metrics
import PCA as p

# plot graph
def plot(data, title, color):
    plt.scatter(range(len(data)), data, c=color)
    plt.title(title)
    plt.xlabel("Time Series Data")
    plt.ylabel("Feature Values")
    plt.grid(True)
    plt.savefig(title)
    plt.show()


# get Feature mattrix for new test data
def get_test_feature_matrix(filename, PCA):
    # import pdb; pdb.set_trace()
    data = parse_and_interpolate_test(filename)[0]
    fft_features = get_fft_features(data)
    entropy_feature = get_entropy(data)
    moving_avg_features = moving_avg(data)
    normal_skew_feature = normal_skew(data)

    feature_mattrix = np.concatenate((moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)

    # get PCA mattrix
    feature_mattrix = PCA.usePCA(feature_mattrix)
    return feature_mattrix


# get Feature mattrix for given data
def get_feature_mattrix():
    meal_files = [
        'MealNoMealData/mealData1.csv', 'MealNoMealData/mealData2.csv',
        'MealNoMealData/mealData3.csv', 'MealNoMealData/mealData4.csv',
        'MealNoMealData/mealData5.csv',
    ]


    meal_data = parse_and_interpolate(meal_files)


    data = meal_data[0]
    fft_features = get_fft_features(data)
    entropy_feature = get_entropy(data)
    moving_avg_features = moving_avg(data)
    normal_skew_feature = normal_skew(data)

    for index in range(1, len(meal_data)):
        data = meal_data[index]

        fft_features = np.concatenate((fft_features, get_fft_features(data)), axis=0)
        moving_avg_features = np.concatenate((moving_avg_features, moving_avg(data)), axis=0)
        entropy_feature = np.concatenate((entropy_feature, get_entropy(data)), axis=0)
        normal_skew_feature = np.concatenate((normal_skew_feature, normal_skew(data)), axis=0)

    feature_mattrix = np.concatenate((moving_avg_features, entropy_feature, fft_features, normal_skew_feature), axis=1)
    np.set_printoptions(suppress=True)

    PCA = p.cal_PCA()
    feature_mattrix = PCA.performPCA(feature_mattrix)

    return feature_mattrix, PCA



def training(feat_mat):
    models = []

    k = km.K_means(10)
    k_labels, k_centroids, k_inertia = k.train(feat_mat)

    d = db.Dbscan(2, 3)
    d_labels = d.train(feat_mat)

    models.append(k)
    models.append(d)

    print ("\nMeal Data: kmeans clustering results") #labels
    print(k_labels)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(feat_mat, k_labels))
    print("Sum of squared distances of samples to their closest cluster center %0.3f"
          % k_inertia)



    print ("\nMeal Data: dbscan cluster results") #labels
    print(d_labels)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(feat_mat, d_labels))



    return models


def testing(models, feat_mat):
    evals = []

    k = models[0]
    d = models[1]

    evals.append(k.test(feat_mat))
    evals.append(d.test(feat_mat))
    return tuple(evals)


def main():
    # get feature mattrix for the given dataset
    feature_mattrix, PCA = get_feature_mattrix()

    # train on the given dataset
    models = training(feature_mattrix)



    filename = "1"
    # input data file and get feature mattrix for new data
    while True:

        filename = input("Input filename (Enter 0 to exit): ")
        if filename == "0":
            break
        try:
            test_data = get_test_feature_matrix([filename], PCA)
        except Exception as e:
            print("Error Cannot open file")
            continue

        # predict new data using learned models
        result = testing(models, test_data)
        result = np.array(result)
        result_labels = ["\nK_MEANS:\n", "\nDBSCAN:\n"]
        for i in range(len(result_labels)):
            print(result_labels[i], result[i])




main()