
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#%% Activation functions

xrange = np.linspace(-2, 2, 200)
plt.figure(figsize=(7,6))
plt.plot(xrange, np.maximum(xrange, 0), label = 'relu')
plt.plot(xrange, np.tanh(xrange), label = 'tanh')
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label = 'logistic')
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output')
plt.show()

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

#%% Exp1
clf = MLPClassifier()

#%% Exp2
clf = MLPClassifier(activation="identity")

#%% Exp3
clf = MLPClassifier(activation="logistic")

#%% Exp4
clf = MLPClassifier(activation="tanh")

#%% Exp5
clf = MLPClassifier(solver="lbfgs")

#%% Exp6
clf = MLPClassifier(solver="sgd")

#%% Exp7
clf = MLPClassifier(learning_rate_init=0.01)

#%% Exp8
clf = MLPClassifier(learning_rate_init=0.0001)

#%% Exp9
clf = MLPClassifier(learning_rate="invscaling")

#%% Exp10
clf = MLPClassifier(learning_rate="adaptive")

#%% Scores original data
clf.fit(ft_train, l_train)
print('Accuracy of ANN on training set: {:.2f}'
     .format(clf.score(ft_train, l_train)))
print('Accuracy of ANN on test set: {:.2f}'
     .format(clf.score(ft_test, l_test)))

#%% Cross Validation

import warnings
warnings.filterwarnings("ignore")

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
