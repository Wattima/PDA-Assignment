import pandas as pd

df = pd.read_csv('C:/Daryl/PDA-Assignment/online_learning_2026.csv')

print(df.info())

# Count missing values in each column
print(df.isnull().sum())

print(df.describe())