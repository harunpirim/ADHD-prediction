#%%
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd

correlation_data = pd.read_csv("correlations.csv", index_col=0) 
plt.figure(figsize=(13,11))
sns.heatmap(correlation_data, cmap="Greens")
plt.title("Correlations between factors", fontsize=25)
plt.savefig('correlations.png', transparent=True)