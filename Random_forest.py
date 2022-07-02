#%% Import libraries and data

from sklearn.model_selection import train_test_split
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