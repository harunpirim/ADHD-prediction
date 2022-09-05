#%% Import libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

#%% PCA

#Load the data
data = pd.read_csv("data_before_pca.csv") 
labels = data.iloc[:,0]
features = data.iloc[:,1:]

#Standarzing the data
scaler = StandardScaler()
scaler.fit(features)
st_data = scaler.transform(features)

#%% PCA for data visualization

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(features)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, labels], axis = 1)

fig = plt.figure(figsize = (6,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_title('2 component PCA', fontsize = 15)
targets = [0,1]
colors = ['g','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['group'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#%% PCA to Speed-up Machine Learning Algorithms
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
