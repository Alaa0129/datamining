# %%
# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

# %%
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    print(nRow, nCol)
    columnNames = list(df)
    nGraphRow = int(nCol / nGraphPerRow)
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# %%
df_por = pd.read_csv('Student_Alcohol_Consumption_Data/student-por.csv')
# %%
# Load the data
df_mat = pd.read_csv('Student_Alcohol_Consumption_Data/student-mat.csv')
df_mat.head()

# %%
df = pd.concat([df_mat, df_por]).drop_duplicates(subset=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'failures', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']).reset_index(drop=True)
df.head()

# see how many rows
print(df.shape)

# %% 
data = df[['address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famsup', 'internet', 'famrel']]
plotPerColumnDistribution(data, 12, 3)

# %%
# scatterplot of Fjob, Mjob, and G3 (highest correlation scores)
for variable in ['Fjob', 'Mjob']:
    x = df[variable].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

# %%
# scatterplot of internet and G3 
for variable in ['internet']:
    x = df[variable].replace(['no', 'yes'], [0, 1])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*1000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

# %%
# scatterplot of famsup (family support) and G3
for variable in ['famsup']:
    x = df[variable].replace(['no', 'yes'], [0, 1])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*1000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

# %%
# scatterplot of guardian and G3
for variable in ['guardian']:
    x = df[variable].replace(['mother', 'father', 'other'], [0, 1, 2])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*1000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

#%% 
# scatterplot of famsize and G3
for variable in ['famsize']:
    x = df[variable].replace(['GT3','LE3'], [0, 1])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*1000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

#%%
# scatterplot of Pstatus (cohabitation status) and G3
for variable in ['Pstatus']:
    x = df[variable].replace(['A', 'T'], [0, 1])
    y = df['G3']

    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 5))
    plt.scatter(df[variable], df['G3'], s=density*1000, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Density')
    plt.title(f'{variable} vs Final Grade')
    plt.show()

# %%
# transform all the categorical variables to numerical values
df['school'] = df['school'].apply(lambda x: 1 if x == 'GP' else 0)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'F' else 0)
df['address'] = df['address'].apply(lambda x: 1 if x == 'U' else 0)
df['famsize'] = df['famsize'].apply(lambda x: 1 if x == 'LE3' else 0)
df['Pstatus'] = df['Pstatus'].apply(lambda x: 1 if x == 'T' else 0)
df['schoolsup'] = df['schoolsup'].apply(lambda x: 1 if x == 'yes' else 0)
df['famsup'] = df['famsup'].apply(lambda x: 1 if x == 'yes' else 0)
df['paid'] = df['paid'].apply(lambda x: 1 if x == 'yes' else 0)
df['activities'] = df['activities'].apply(lambda x: 1 if x == 'yes' else 0)
df['nursery'] = df['nursery'].apply(lambda x: 1 if x == 'yes' else 0)
df['higher'] = df['higher'].apply(lambda x: 1 if x == 'yes' else 0)
df['internet'] = df['internet'].apply(lambda x: 1 if x == 'yes' else 0)
df['romantic'] = df['romantic'].apply(lambda x: 1 if x == 'yes' else 0)

# %%
# transform nominal variables to numerical values
df['Mjob'] = df['Mjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
df['Fjob'] = df['Fjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
df['reason'] = df['reason'].replace(['home', 'reputation', 'course', 'other'], [0, 1, 2, 3])
df['guardian'] = df['guardian'].replace(['mother', 'father', 'other'], [0, 1, 2])

# %%  plot the distribution of the target variable G3 before categorization
plt.figure(figsize=(10, 5))
sns.histplot(df['G3'])
plt.show()

# %% mean and std before categorization
mean = df.mean()
std = df.std()

print(mean)
print(std)

# %%
# Categorize G3 into 4 categories based on the distribution
# find distribution of G3

df['G3'] = df['G3'].apply(lambda x: 1 if x <= 8 else 2 if x <= 10 else 3 if x <= 13 else 4)

#%%
count_of_value = (df['G3'] == 4).sum()
print(f'Count of value : {count_of_value}')

# %% 
# find the mean and standard deviation of each column
mean = df.mean()
std = df.std()

print(mean)
print(std)
# %%
# plot the distribution of the target variable G3 after categorization
plt.figure(figsize=(10, 5))
sns.histplot(df['G3'])
plt.show()

# %%
# plot the distribution of all the features in the dataset
df.hist(figsize=(20, 20))
plt.show()

# %% 
# find value counts of the distribution of all features
for col in df.columns:
    print(col)
    print(df[col].value_counts(normalize=True))
    print()

# %% 
# remove entries where Medu =0 and Fedu = 0, as there are not many entries with these values
df = df[df['Fedu'] != 0]
df = df[df['Medu'] != 0]

# %%
# export the data
df.to_csv('Student_Alcohol_Consumption_Data/student-transformed.csv', index=False)

# %%
df.head()
# %%
