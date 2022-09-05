#%% Libraries
from cv2 import rotate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
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

#%%Exp1
clf = XGBClassifier(seed=0)

#%%Exp2
clf = XGBClassifier(learning_rate = 0.1, seed=0)

#%%Exp3
clf = XGBClassifier(learning_rate = 0.01, seed=0)

#%%Exp4
clf = XGBClassifier(learning_rate = 1, n_estimators = 2, seed=0)

#%%Exp5
clf = XGBClassifier(learning_rate = 0.1, n_estimators = 200, seed=0)

#%%Exp6
clf = XGBClassifier(learning_rate = 0.1, max_depth = 3, n_estimators = 200, seed=0)

#%%Exp7
clf = XGBClassifier(learning_rate = 1, max_depth = 10, n_estimators = 10, seed=0)

#%% Scores 
clf.fit(ft_train, l_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(ft_train, l_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(ft_test, l_test)))

#%% Feature importance
import matplotlib.pyplot as plt

clf.fit(ft_train, l_train)
feature_names = list(original_data.columns.values)
importances = clf.feature_importances_

plt.figure(figsize=(8,5))
plt.style.use('seaborn-darkgrid')
plt.bar(feature_names[1:],importances,color="lightcoral")
plt.title("Feature importances")
plt.ylabel("Importance")
plt.xticks(rotation="90")
plt.show()

#%% Cross Validation

#Accuracy
print("Accuracy")
cv = cross_val_score(clf,o_features,o_labels)
#cv2 = cross_val_score(clf,pca_features,pca_labels)
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Precision
print("Precision")
cv = cross_val_score(clf,o_features,o_labels, scoring="precision")
#cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="precision")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Recall
print("Recall")
cv = cross_val_score(clf,o_features,o_labels, scoring="recall")
#cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="recall")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))

#Area under the curve
print("AUC")
cv = cross_val_score(clf,o_features,o_labels, scoring="roc_auc")
#cv2 = cross_val_score(clf,pca_features,pca_labels, scoring="roc_auc")
print("original:",round(np.mean(cv),3))
#print("pca:",round(np.mean(cv2),3))
