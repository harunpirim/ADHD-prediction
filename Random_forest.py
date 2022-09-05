#%% Import libraries and data

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
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
clf = DecisionTreeClassifier(random_state=42)

#%%Exp2
clf = DecisionTreeClassifier(max_depth = 20, random_state=42)

#%%Exp3
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)

#%%Exp4
clf = DecisionTreeClassifier(criterion="log_loss", class_weight="balanced", random_state=42)

#%%Exp5
clf = DecisionTreeClassifier(class_weight="balanced", random_state=42)

#%% Scores 

clf.fit(ft_train, l_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(ft_train, l_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(ft_test, l_test)))

#%% Cross Validation

cv = cross_val_score(clf,o_features,o_labels)
cv2 = cross_val_score(clf,pca_features,pca_labels)
print("original:",round(np.mean(cv),3))
print("pca:",round(np.mean(cv2),3))

# Random Forest

#%% Exp1
clf = RandomForestClassifier(random_state=0)

#%% Exp2
clf = RandomForestClassifier(n_estimators=10, random_state=0)

#%% Exp3
clf = RandomForestClassifier(n_estimators=100, random_state=0)

#%% Exp4
clf = RandomForestClassifier(criterion="entropy", random_state=0)

#%% Exp5
clf = RandomForestClassifier(criterion="log_loss", random_state=0)

#%% Exp6
clf = RandomForestClassifier(class_weight="balanced", random_state=0)

#%% Exp7
clf = RandomForestClassifier(max_depth=100, random_state=0)

#%% Feature importance
import matplotlib.pyplot as plt

clf.fit(ft_train, l_train)
feature_names = list(original_data.columns.values)
print(feature_names)
importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=feature_names[1:])
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
#std = np.std(importances)

fig, ax = plt.subplots(figsize=(7,5))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.figure(figsize=(8,5))
plt.style.use('seaborn-darkgrid')
plt.bar(feature_names[1:],importances,color="forestgreen")
plt.title("Feature importances")
plt.ylabel("Importance")
plt.xticks(rotation="90")
plt.show()

#%% Scores original data

clf.fit(ft_train, l_train)
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(clf.score(ft_train, l_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(clf.score(ft_test, l_test)))

#%% Cross Validation

clf.fit(ft_train, l_train)

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