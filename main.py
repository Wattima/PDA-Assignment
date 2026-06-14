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

# Removing in progress students as we don't know the outcome
df = df[df['completion_status'] != 'In Progress'].copy()

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
df['completion_status'] = df['completion_status'].map({'Dropped': 0, 'Completed': 1})

# Correlation matrix heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Generate the heatmap
# Satisfaction rating shows high correlation
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Student Features')
plt.clf()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# defining features and target
X = df[['hours_per_week', 'satisfaction_rating']]
y = df['completion_status']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling X data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)

# evaluate
# very high score of 0.98. Same score across
score = svm_model.score(X_test_scaled, y_test)
print(f"Model Accuracy on Test Set: {score:.2f}")
print(f"Training Accuracy: {svm_model.score(X_train_scaled, y_train):.2f}")
print(f"Test Accuracy: {svm_model.score(X_test_scaled, y_test):.2f}")

# Visualization
# Plotting the test set and the decision boundary
df_scaled = pd.DataFrame(X_test_scaled, columns=['Hours', 'Satisfaction'])
df_scaled['Status'] = y_test.values

# Use Seaborn's scatterplot with hue (color)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scaled, x='Hours', y='Satisfaction', hue='Status', palette='coolwarm', style=None, s=100)

# adding hyperplane
w = svm_model.coef_[0]
b = svm_model.intercept_[0]

x_range = np.array([df_scaled['Hours'].min(), df_scaled['Hours'].max()])
y_range = -(w[0] / w[1]) * x_range - (b / w[1])

plt.plot(x_range, y_range, 'k--', linewidth=2, label='Decision Boundary')
plt.title('SVM Classification: Student Outcomes')
plt.legend()
plt.show()


