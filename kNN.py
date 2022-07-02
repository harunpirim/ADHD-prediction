#%% Import libraries and data

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pandas as pd

original_data = pd.read_csv("data_before_pca.csv")
o_features = original_data.iloc[:,1:]
o_labels = original_data.iloc[:,0]

ft_train_o, ft_test_o, l_train_o, l_test_o = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)
ft_val_o, ft_test_o, l_val_o, l_test_o = train_test_split(ft_test_o,l_test_o,test_size=0.5, random_state=42)

pca_data = pd.read_csv("data_after_pca.csv")
pca_features = pca_data.iloc[:,1:]
pca_labels = original_data.iloc[:,0]
    
ft_train_pca, ft_test_pca, l_train_pca, l_test_pca = train_test_split(pca_features, pca_labels, test_size=0.3, random_state=42)
ft_val_pca, ft_test_pca, l_val_pca, l_test_pca = train_test_split(ft_test_pca,l_test_pca,test_size=0.5, random_state=42)

print("Data uploaded successfully")

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
train_o = pd.concat([ft_train_o, l_train_o], axis = 1)
predictions_o=knn(train_o,ft_val_o,5)

cm = metrics.confusion_matrix(l_val_o, predictions_o)
acc = metrics.accuracy_score(l_val_o, predictions_o)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'PuRd_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(acc,2))
plt.title(all_sample_title, size = 15)
    
#%% KNN PCA data 
train_pca = pd.concat([ft_train_pca, l_train_pca], axis = 1)
predictions_pca=knn(train_pca,ft_val_pca,5)

cm = metrics.confusion_matrix(l_val_pca, predictions_pca)
acc = metrics.accuracy_score(l_val_pca, predictions_pca)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'PuRd_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(acc,2))
plt.title(all_sample_title, size = 15)

#%% Sensitivity of kNN to k

k_range = range(1,15)
scores_o = []
scores_pca = []

for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k, weights='distance')
    clf.fit(ft_train_o, l_train_o)
    scores_o.append(clf.score(ft_val_o, l_val_o))
    clf.fit(ft_train_pca, l_train_pca)
    scores_pca.append(clf.score(ft_val_pca, l_val_pca))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores_o, label="original_data")
plt.scatter(k_range, scores_pca, label="pca_data")
plt.legend()

#%% Sensitivity of kNN to k

k_range = range(1,15)
scores_o = []
scores_pca = []

for k in k_range:
    predictions_o=knn(train_o,ft_val_o,k)
    scores_o.append(metrics.accuracy_score(l_val_o, predictions_o))
    predictions_pca=knn(train_pca,ft_val_pca,k)
    scores_pca.append(metrics.accuracy_score(l_val_pca, predictions_pca))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores_o, label="original_data")
plt.scatter(k_range, scores_pca, label="pca_data")
plt.legend()