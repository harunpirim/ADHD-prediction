#%% Import libraries and data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
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

#%% Regresion model for original data

#Instance of the Model
logisticRegr = LogisticRegression(solver = 'lbfgs')

#Training the model to learn the relationship between features and labels
logisticRegr.fit(ft_train_o, l_train_o)

#Predict labels for valid set (new information)
predictions = logisticRegr.predict(ft_val_o)
cm = metrics.confusion_matrix(l_val_o, predictions)
score_val = logisticRegr.score(ft_val_o, l_val_o)
score_train = logisticRegr.score(ft_train_o, l_train_o)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score_val,2))
plt.title(all_sample_title, size = 15)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(score_train))
print('Accuracy of Logistic regression classifier on validation set: {:.2f}'
     .format(score_val))

#%% Regresion model for pca data

#Instance of the Model
logisticRegr = LogisticRegression(solver = 'lbfgs')

#Training the model to learn the relationship between features and labels
logisticRegr.fit(ft_train_pca, l_train_pca)

#Predict labels for valid set (new information)
predictions = logisticRegr.predict(ft_val_pca)
cm = metrics.confusion_matrix(l_val_pca, predictions)
score_val = logisticRegr.score(ft_val_pca, l_val_pca)
score_train = logisticRegr.score(ft_train_pca, l_train_pca)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score_val,2))
plt.title(all_sample_title, size = 15)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(score_train))
print('Accuracy of Logistic regression classifier on validation set: {:.2f}'
     .format(score_val))