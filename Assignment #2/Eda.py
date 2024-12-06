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
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    name = 'Family Characteristics'
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {name}', fontsize=15)
    plt.show()

# %% define list of variables to check for outliers, based on family characteristics
variables = ['address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famsup', 'internet', 'famrel']

data = df[['address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famsup', 'internet', 'famrel', 'G3']]
plotCorrelationMatrix(data, 8)

# %%
none_binary_variables = ['Medu', 'Fedu', 'traveltime', 'famrel', 'Mjob', 'Fjob']
# visualize outliers for variables using a quantile-based boxplot
for column in none_binary_variables:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()
# %% 
# show outlier for variables 
for variable in variables:
    tmp = df[variable]
    z_scores = stats.zscore(tmp)
    outlier_masks = z_scores > 3
    plt.scatter(np.arange(len(tmp)), tmp, c=outlier_masks, cmap='coolwarm')
    plt.title(variable)
    plt.show()

#%%
# remove outliers, if relevant
#filtered_data = tmp[~outlier_masks]
#df['Medu'] = filtered_data

# %%
# Look at pearson correlation between Fedu, Medu, and internet, and G3
for variable in ['Fedu', 'Medu', 'internet']:
    pearson_correlation = df[variable].corr(df['G3'])
    print(f'Pearson correlation between {variable} and G3: {pearson_correlation}')
    plt.figure(figsize=(10, 5))
    sns.regplot(x=variable, y='G3', data=df, color='blue', line_kws={'color': 'red'}, label='best-fit line')
    plt.xlabel(variable)
    plt.ylabel('Final Grade')
    plt.title(f'{variable} vs Final Grade')
    plt.legend()
    plt.show()

# %%
# calculate the pearson correlation coefficient between the variables
pearson_correlation = df['Medu'].corr(df['G3'])
pearson_correlation

# %%
df.drop('G3', axis=1).corrwith(df.G3).sort_values().plot(kind='barh', figsize=(15, 10))