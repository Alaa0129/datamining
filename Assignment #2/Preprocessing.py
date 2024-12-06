# %%
# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
df['Mjob'] = df['Mjob'].replace(['teacher', 'health', 'civil', 'at_home', 'other', 'services'], [0, 1, 2, 3, 4, 5])
df['Fjob'] = df['Fjob'].replace(['teacher', 'health', 'civil', 'at_home', 'other', 'services'], [0, 1, 2, 3, 4, 5])
df['reason'] = df['reason'].replace(['home', 'reputation', 'course', 'other'], [0, 1, 2, 3])
df['guardian'] = df['guardian'].replace(['mother', 'father', 'other'], [0, 1, 2])

# %%  plot the distribution of the target variable G3 before categorization
plt.figure(figsize=(10, 5))
sns.histplot(df['G3'])
plt.show()

# %% mean and std before categorization
mean = df.mean()
std = df.std()

#print(mean)
print(std)
# %%
# Categorize G3 into 4 categories
df['G3'] = df['G3'].apply(lambda x: 1 if x <= 5 else 2 if x <= 10 else 3 if x <= 15 else 4)

# %% 
# find the mean and standard deviation of each column
mean = df.mean()
std = df.std()

#print(mean)
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
# export the data
df.to_csv('Student_Alcohol_Consumption_Data/student-transformed.csv', index=False)

# %%
