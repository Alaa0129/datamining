# %%
# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow, title = None):
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
        plt.title(f'{title} {columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# %% Return the label for current dataset
def returnGender():
    return 'Male' if data is df_boys else 'Female' if data is df_girls else 'Combined'


# %% load the dataset for students studying portuguese
df_por = pd.read_csv('Student_Alcohol_Consumption_Data/student-por.csv')
# %% load the dataset for students studying math
df_mat = pd.read_csv('Student_Alcohol_Consumption_Data/student-mat.csv')

# %% combine the two datasets and remove duplicates
df = pd.concat([df_mat, df_por]).drop_duplicates(subset=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'failures', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']).reset_index(drop=True)
df.head()

# see how many rows
print(df.shape)

# %% extract the features related to family relations
df = df[['sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'famsup', 'famrel', 'G3']]

#Create two dataframes cossistent of only male and female students
df_boys = df[df['sex'] == 'M']
df_girls = df[df['sex'] == 'F']
#Drop Sex feature as it is not family related
df_boys = df_boys.drop(['sex'], axis=1)
df_girls = df_girls.drop(['sex'], axis=1)
#combine dataframes into an array of dataframes
dataframes = [df_boys, df_girls, df]

#print the column distribution for each dataframe
plotPerColumnDistribution(df.drop(['sex'], axis=1), 12, 3)
plotPerColumnDistribution(df_boys, 12, 3, 'Male')
plotPerColumnDistribution(df_girls, 12, 3, 'Female')

# %%
# scatterplot of Fjob, Mjob, and G3 (highest correlation scores)
for data in dataframes:
    for variable in ['Fjob', 'Mjob']:
        x = data[variable].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
        y = data['G3']

        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 5))
        plt.scatter(data[variable], data['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Density')
        plt.title(f' {returnGender()} {variable} vs Final Grade')
        plt.show()

# %%
# scatterplot of famsup (family support) and G3
for data in dataframes:
    for variable in ['famsup']:
        x = data[variable].replace(['no', 'yes'], [0, 1])
        y = data['G3']

        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 5))
        plt.scatter(data[variable], data['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Density')
        plt.title(f'{returnGender()} {variable} vs Final Grade')
        plt.show()

# %%
# scatterplot of guardian and G3
for data in dataframes:
    for variable in ['guardian']:
        x = data[variable].replace(['mother', 'father', 'other'], [0, 1, 2])
        y = data['G3']

        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 5))
        plt.scatter(data[variable], data['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Density')
        plt.title(f'{returnGender()} {variable} vs Final Grade')
        plt.show()

#%% 
# scatterplot of famsize and G3
for data in dataframes:
    for variable in ['famsize']:
        x = data[variable].replace(['GT3','LE3'], [0, 1])
        y = data['G3']

        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 5))
        plt.scatter(data[variable], data['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Density')
        plt.title(f'{returnGender()} {variable} vs Final Grade')
        plt.show()

#%% 
# scatter plot for address and G3
for data in dataframes:
    for variable in ['address']:
        x = data[variable].replace(['U', 'R'], [0, 1])
        y = data['G3']

        xy = np.vstack([x, y])
        density = stats.gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 5))
        plt.scatter(data[variable], data['G3'], s=density*5000, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Density')
        plt.title(f'{returnGender()} {variable} vs Final Grade')
        plt.show()

# %%
# transform all the categorical variables to numerical values
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'F' else 0)
df['address'] = df['address'].apply(lambda x: 1 if x == 'U' else 0)
df['famsize'] = df['famsize'].apply(lambda x: 1 if x == 'LE3' else 0)
df['Pstatus'] = df['Pstatus'].apply(lambda x: 1 if x == 'T' else 0)
df['famsup'] = df['famsup'].apply(lambda x: 1 if x == 'yes' else 0)

# %%
# transform nominal variables to numerical values
df['Mjob'] = df['Mjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
df['Fjob'] = df['Fjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [0, 1, 2, 3, 4])
df['guardian'] = df['guardian'].replace(['mother', 'father', 'other'], [0, 1, 2])


# %%
# Categorize G3 into 4 categories based on the distribution
# find distribution of G3

df['G3'] = df['G3'].apply(lambda x: 1 if x <= 8 else 2 if x <= 10 else 3 if x <= 13 else 4)

#%% Check the count for data entries for each interval
count_of_value = (df['G3'] == 4).sum()
print(f'Count of value : {count_of_value}')


# %% 
# remove entries where Medu =0 and Fedu = 0, as there are not many entries with these values
df = df[df['Fedu'] != 0]
df = df[df['Medu'] != 0]

# %% Again set dataframes
df_boys = df[df['sex'] == 0]
df_girls = df[df['sex'] == 1]
df_boys = df_boys.drop(['sex'], axis=1)
df_girls = df_girls.drop(['sex'], axis=1)
dataframes = [df_boys, df_girls, df]

# %% visualize the G3 distribution
for data in dataframes:
    plt.figure(figsize=(10, 5))
    sns.histplot(data['G3'])
    plt.title(f'{returnGender()} G3 dist')
    plt.show()
    
# %%
# plot the distribution of all the features in the dataset
for data in dataframes:
    data.hist(figsize=(20, 20))
    plt.title(returnGender())
    plt.show()

# %% 
# find value counts of the distribution of all features
for data in dataframes:
    for col in data.columns:
        print(col)
        print(data[col].value_counts(normalize=True))
        print(returnGender())
        print()

# %%
# export the data
df.to_csv('Student_Alcohol_Consumption_Data/student-transformed.csv', index=False)

# %% shows df head
df.head()
# %%