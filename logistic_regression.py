#%% Import libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np

#%% Original data
original_data = pd.read_csv("data_before_pca.csv")
o_features = original_data.iloc[:,1:]
o_labels = original_data.iloc[:,0]
ft_train, ft_test, l_train, l_test = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)
print("Original Data uploaded successfully")

#%% Normalized original data
scaler = MinMaxScaler()
o_features = scaler.fit_transform(o_features)
ft_train, ft_test, l_train, l_test = train_test_split(o_features, o_labels, test_size=0.3, random_state=42)
print("Normalized original data uploaded successfully")

#%% PCA 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(o_features)
two_pca = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
ft_train, ft_test, l_train, l_test = train_test_split(two_pca, o_labels, test_size=0.3, random_state=42)
print("2 PCA Data uploaded successfully")

#%% PCA
pca_data = pd.read_csv("data_after_pca.csv")
pca_features = pca_data.iloc[:,1:]
pca_labels = original_data.iloc[:,0]
ft_train, ft_test, l_train, l_test_pca = train_test_split(pca_features, pca_labels, test_size=0.3, random_state=42)

print("PCA Data uploaded successfully")

#%% Exp1
clf = LogisticRegression()

#%% Exp2
clf = LogisticRegression(penalty="l1", solver="liblinear")

#%% Exp3
clf = LogisticRegression(penalty="none")

#%% Exp4
clf = LogisticRegression(C=0.1)

#%% Exp5
clf = LogisticRegression(C=100)

#%% Exp6
clf = LogisticRegression(C=0.1, class_weight="balanced")

#%% Scores

clf.fit(ft_train, l_train)
predictions = clf.predict(ft_test)
cm = metrics.confusion_matrix(l_test, predictions)
score_train = clf.score(ft_train, l_train)
score_test = clf.score(ft_test, l_test)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(score_train))
print('Accuracy of Logistic regression classifier on validation set: {:.2f}'
     .format(score_test))

#%% Confusion matrix

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score_test,2))
plt.title(all_sample_title, size = 15)

#%% Crossvalidation

#Accuracy
print("Accuracy")
cv = cross_val_score(clf,o_features,o_labels)
cv2 = cross_val_score(clf,pca_features,pca_labels)
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Precision
print("Precision")
cv = cross_val_score(clf,o_features,o_labels, scoring="precision")
cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="precision")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Recall
print("Recall")
cv = cross_val_score(clf,o_features,o_labels, scoring="recall")
cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="recall")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Area under the curve
print("AUC")
cv = cross_val_score(clf,o_features,o_labels, scoring="roc_auc")
cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="roc_auc")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

