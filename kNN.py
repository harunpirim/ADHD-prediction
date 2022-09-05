#%% Import libraries and data

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pandas as pd
import numpy as np

#%% Original data
original_data = pd.read_csv("data_before_pca.csv")
o_features = original_data.iloc[:,1:]
o_labels = original_data.iloc[:,0]
ft_train, ft_test, l_train, l_test = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)

print("Original data uploaded successfully")

#%% Normalized original data
scaler = MinMaxScaler()
o_features = scaler.fit_transform(o_features)
ft_train, ft_test, l_train, l_test = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)
print("Normalized original data uploaded successfully")


#%% PCA data
pca_data = pd.read_csv("data_after_pca.csv")
pca_features = pca_data.iloc[:,1:]
pca_labels = original_data.iloc[:,0]
ft_train, ft_test, l_train, l_test = train_test_split(pca_features, pca_labels, test_size=0.3, random_state=42)

print("PCA data uploaded successfully")

#%% KNN

def knn(train, valid, k):

    predictions=[]

    for new_child in valid.values:

        neighbor_distances_and_indices = []

        for index, child in enumerate(train.values):
            # Calculate the distance between the query example and the current example from the data.
            dist = distance.euclidean(child[:-1], new_child)
            
            # Add the distance and the index of the example to an ordered collection
            neighbor_distances_and_indices.append((dist, index))
        
        # Sort the ordered collection of distances and indices from
        # smallest to largest (in ascending order) by the distances
        sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
        
        # Pick the first K entries from the sorted collection
        k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

        # Get the labels of the selected K entries
        k_nearest_labels = [train.values[i][-1] for distance, i in k_nearest_distances_and_indices]
        predictions.append(statistics.mode(k_nearest_labels))

    return predictions

#%% KNN original data
train_o = pd.concat([ft_train, l_train], axis = 1)
predictions_o=knn(train_o,ft_test,5)

cm = metrics.confusion_matrix(l_test, predictions_o)
acc = metrics.accuracy_score(l_test, predictions_o)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'PuRd_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(acc,2))
plt.title(all_sample_title, size = 15)
    
#%% KNN PCA data 
train_pca = pd.concat([ft_train, l_train], axis = 1)
predictions_pca=knn(train_pca,ft_test,5)

cm = metrics.confusion_matrix(l_test, predictions_pca)
acc = metrics.accuracy_score(l_test, predictions_pca)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'PuRd_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(acc,2))
plt.title(all_sample_title, size = 15)

#%% Sensitivity of kNN to k

k_range = range(1,15)
scores_d = []
scores_u = []

for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k, weights='distance')
    clf.fit(ft_train, l_train)
    scores_d.append(clf.score(ft_test, l_test))
    clf2 = KNeighborsClassifier(n_neighbors = k, weights='uniform')
    clf2.fit(ft_train, l_train)
    scores_u.append(clf2.score(ft_test, l_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores_d, label="distance")
plt.scatter(k_range, scores_u, label="uniform")
plt.legend()
plt.show()


#%% Cross Validation

k_range=[1,3,5,10]
means = []
means2 = []

for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k, weights='uniform')
    cv = cross_val_score(clf,o_features,o_labels, scoring="roc_auc")
    means.append(round(np.mean(cv),3))
    #cv2 = cross_val_score(clf,pca_features,pca_labels)
    #means2.append(np.mean(cv2))

print(means)
#print(means2)

