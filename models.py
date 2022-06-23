#%% Import libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

#%% PCA

#Load the data
data = pd.read_csv("data_before_pca.csv") 
labels = data.iloc[:,0]
features = data.iloc[:,1:]

#Standarzing the data
scaler = StandardScaler()
scaler.fit(features)
st_data = scaler.transform(features)

#PCA implementation
pca = PCA(.95)
pca.fit(st_data)
pca_data = pca.transform(st_data)

#Data exportation
df = pd.DataFrame(pca_data)
df.to_csv('data_after_pca.csv')

#%% Data split

original_data = pd.read_csv("data_before_pca.csv")
o_features = original_data.iloc[:,1:]
o_labels = original_data.iloc[:,0]

ft_train_o, ft_test_o, l_train_o, l_test_o = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)
ft_val_o, ft_test_o, l_val_o, l_test_o = train_test_split(ft_test_o,l_test_o,test_size=0.5, random_state=42)

d = {"Train": [l_train_o.sum(),len(l_train_o)-l_train_o.sum(),len(l_train_o)],
"Valid": [l_val_o.sum(),len(l_val_o)-l_val_o.sum(),len(l_val_o)],
"Test": [l_test_o.sum(),len(l_test_o)-l_test_o.sum(),len(l_test_o)]}

print("--------------Original data--------------")
print ("{:<8} {:<15} {:<10} {:<10}".format('Set','ADHD','Control','Total'))
for k, v in d.items():
    adhd, control, total = v
    print ("{:<8} {:<15} {:<10} {:<10}".format(k, adhd, control, total))

pca_data = pd.read_csv("data_after_pca.csv")
pca_features = pca_data.iloc[:,1:]
pca_labels = original_data.iloc[:,0]
    
ft_train_pca, ft_test_pca, l_train_pca, l_test_pca = train_test_split(pca_features, pca_labels, test_size=0.3, random_state=42)
ft_val_pca, ft_test_pca, l_val_pca, l_test_pca = train_test_split(ft_test_pca,l_test_pca,test_size=0.5, random_state=42)

d = {"Train": [l_train_pca.sum(),len(l_train_pca)-l_train_pca.sum(),len(l_train_pca)],
"Valid": [l_val_pca.sum(),len(l_val_pca)-l_val_pca.sum(),len(l_val_pca)],
"Test": [l_test_pca.sum(),len(l_test_pca)-l_test_pca.sum(),len(l_test_pca)]}

print("\n")
print("-----------------PCA data-----------------")
print ("{:<8} {:<15} {:<10} {:<10}".format('Set','ADHD','Control','Total'))
for k, v in d.items():
    adhd, control, total = v
    print ("{:<8} {:<15} {:<10} {:<10}".format(k, adhd, control, total))

#%% Regresion model for original data

#Instance of the Model
logisticRegr = LogisticRegression(solver = 'lbfgs')

#Training the model to learn the relationship between features and labels
logisticRegr.fit(ft_train_o, l_train_o)

#Predict labels for valid set (new information)
predictions = logisticRegr.predict(ft_val_o)
cm = metrics.confusion_matrix(l_val_o, predictions)
score = logisticRegr.score(ft_val_o, l_val_o)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score,2))
plt.title(all_sample_title, size = 15)

#%% Regresion model for pca data

#Instance of the Model
logisticRegr = LogisticRegression(solver = 'lbfgs')

#Training the model to learn the relationship between features and labels
logisticRegr.fit(ft_train_pca, l_train_pca)

#Predict labels for valid set (new information)
predictions = logisticRegr.predict(ft_val_pca)
cm = metrics.confusion_matrix(l_val_pca, predictions)
score = logisticRegr.score(ft_val_pca, l_val_pca)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score,2))
plt.title(all_sample_title, size = 15)

#%% KNN

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def main():
       
    clf_data = [
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]

    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    clf_query = [33]
    clf_k_nearest_neighbors, clf_prediction = knn(
        clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode
    )

if __name__ == '__main__':
    main()