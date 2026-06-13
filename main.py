import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
# pd.set_option('display.max_columns', 200)

# import data
df = pd.read_csv('C:/Daryl/PDA-Assignment/online_learning_2026.csv')

print(df.shape)
print(df.columns)
print(df.head(20))

# Count missing values in each column
print(df.isnull().sum())

print(df.describe())

# check for null values
print("Are there any nulls in the dataset?", df.isna().values.any())

# check for duplicates
print("Total duplicates:", df.duplicated().sum())

# check for balanced classes?
# Potential bias toward 'Completed' as majority class
sns.countplot(x='completion_status', data=df)
plt.title('Distribution of Target Classes')
plt.clf()

# Slight positive correlation
sns.boxplot(x='completion_status', y='self_motivation_score', data=df)
plt.title('Does a higher self-motivation score lead to completion?')
plt.clf()

# encoding completion_status for heatmap
df['completion_status'] = df['completion_status'].map({'Dropped': 0, 'In Progress': 1, 'Completed': 2})

# Correlation matrix heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Generate the heatmap
# Satisfaction rating shows high correlation
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Student Features')
plt.show()