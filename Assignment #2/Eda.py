#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)
df = pd.read_csv('Student_Alcohol_Consumption_Data/student-transformed.csv')

df.head()

df = df.drop('G1', axis=1)
df = df.drop('G2', axis=1)
df = df.drop('failures', axis=1)
# df.drop('G3', axis=1).corrwith(df.G3).sort_values().plot(kind='barh', figsize=(15, 10))

corr = df.corr(method='pearson', numeric_only=True)
corr

fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(corr, vmin= -1, vmax=1, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu_r', annot=True, linewidth=0.5, ax=ax)

# %% 
# check for outliers based on a normal distribution
tmp = df[stats.zscore(df) < 3]
print(tmp)
plt.figure(figsize=(20, 10))
sns.boxplot(data=df, orient='h')
plt.show()

# %%
df.drop('G3', axis=1).corrwith(df.G3).sort_values().plot(kind='barh', figsize=(15, 10))
