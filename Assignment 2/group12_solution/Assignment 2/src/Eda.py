#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from Preprocessing import *
from statsmodels.formula.api import ols

# %%
# Correlation matrix for family characteristics
def plotCorrelationMatrix(df, graphWidth, gender = None):
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
    plt.title(f'{gender} Correlation Matrix for {name}', fontsize=15)
    plt.show()

# %% Functions to return label for current dataset
def returnGender():
    return 'Male' if data is df_boys else 'Female' if data is df_girls else 'Combined'

# %% define list of variables to check for outliers, based on family characteristics
variables = ['address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famsup', 'famrel']

for data in dataframes:
    plotCorrelationMatrix(data, 8, returnGender())

# %%
none_binary_variables = ['Medu', 'Fedu', 'traveltime', 'famrel', 'Mjob', 'Fjob']
# visualize outliers for variables using a quantile-based boxplot
for column in none_binary_variables:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()
# %% 
# show outlier for variables using a scatterplot
for variable in variables:
    tmp = df[variable]
    z_scores = stats.zscore(tmp)
    outlier_masks = z_scores > 3
    plt.scatter(np.arange(len(tmp)), tmp, c=outlier_masks, cmap='coolwarm')
    plt.title(variable)
    plt.show()

# %% Calculate Pearson r and visualize relation in scatter plot with best fit linear regression line.
for data in dataframes:
    for variable in ['Fedu', 'Medu', 'famrel']:
        pearson_correlation = data[variable].corr(data['G3'])
        print(f'Pearson correlation between {variable} and G3: {pearson_correlation}')
        plt.figure(figsize=(10, 5))
        sns.regplot(x=variable, y='G3', data=data, color='blue', line_kws={'color': 'red'}, label='best-fit line')
        plt.xlabel(variable)
        plt.ylabel('Final Grade')
        plt.title(f'{returnGender()} {variable} vs Final Grade')
        plt.legend()
        plt.show()

# %% Visualize the average student grade based on mother and fathers education level
for data in dataframes:
    data.groupby('Medu')['G3'].mean().plot(kind='bar') 
    plt.title(f'{returnGender()} Average Final Grade by Mother Education Level')
    plt.show()

    data.groupby('Fedu')['G3'].mean().plot(kind='bar')
    plt.title(f'{returnGender()} Average Final Grade by Father Education Level')
    plt.show()

# %% Visuaælize correlatio between all features and final grade.
for data in dataframes:
    data.drop('G3', axis=1).corrwith(data.G3).sort_values().plot(kind='barh', figsize=(15, 10))
    plt.title(f'{returnGender()} Correlation with Final Grade')
    plt.show()

# %% Update Perform OLS on Medu and gender with regard to G3

df_boys_p = df_boys[['Medu', 'G3']]
df_girls_p = df_girls[['Medu', 'G3']]
df_boys_p['Gender'] = 0
df_girls_p['Gender'] = 1

df_combined_p = pd.concat([df_boys_p, df_girls_p], ignore_index=True)

formula = 'G3 ~ Medu + Gender + Medu:Gender'
interaction_model = ols(formula, data=df_combined_p).fit()

print(interaction_model.summary())
# %% Update Perform OLS on Medu and gender with regard to G3

df_boys_p = df_boys[['Fedu', 'G3']]
df_girls_p = df_girls[['Fedu', 'G3']]
df_boys_p['Gender'] = 0
df_girls_p['Gender'] = 1

df_combined_p = pd.concat([df_boys_p, df_girls_p], ignore_index=True)

formula = 'G3 ~ Fedu + Gender + Fedu:Gender'
interaction_model = ols(formula, data=df_combined_p).fit()

print(interaction_model.summary())

# %%
