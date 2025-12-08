import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter


file_path = "Titanic-Dataset.csv"
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/titanic-dataset",
  file_path,
)

print("Initial Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())                   
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])    
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

df = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
df = df.drop_duplicates()
print(f"\n {891 - len(df)} duplicates removed , Number of passengers now: {len(df)}")

print("\nCleaned Dataset Info:")
print(df.info())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Explore
print("\nSummary Statistics:")
print(df.describe())

print("\nSurvival Rate (%):")
print((df['Survived'].value_counts(normalize=True) * 100).round(2))

print("\nSurvival by Pclass (%):")
print((df.groupby('Pclass', observed=False)['Survived'].mean() * 100).round(2))

print("\nSurvival by Sex (%):")
print((df.groupby('Sex', observed=False)['Survived'].mean() * 100).round(2))

numeric_df = df.select_dtypes(include=[np.number])
print("\nCorrelation Matrix:")
print(numeric_df.corr().round(3))

# Visualization
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Pclass')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age by Survival')
plt.show()

sns.pairplot(numeric_df, hue='Survived')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
df.head()