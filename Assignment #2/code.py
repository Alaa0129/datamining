# %%
# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
# Load the data
df = pd.read_csv('Student_Alcohol_Consumption_Data/student-mat.csv')
df.head()

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


# %%
# export the data
df.to_csv('Student_Alcohol_Consumption_Data/student-mat-transformed.csv', index=False)