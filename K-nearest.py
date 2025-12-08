import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv("titanic_cleaned.csv")
print(data.head(10))
print(data.shape)

# Count 'male'
print("Number of males:", data.isin(['male']).sum().sum())

# Missing values and duplicates
info = pd.DataFrame(data.isnull().sum(), columns=["IsNull"])
info["IsNa"] = data.isna().sum()
info["Duplicate"] = data.duplicated().sum()
info["Unique"] = data.nunique()
print(info.T)

# Remove duplicates
df = data.drop_duplicates()
print("Shape after dropping duplicates:", df.shape)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Histograms
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
sns.histplot(df['Survived'], kde=True, color='r')
plt.title('Survived Histogram')
plt.subplot(1, 2, 2)
sns.histplot(df['Sex'], kde=True, color='r')
plt.title('Sex Histogram')
plt.show()

# Diagnostic plots function
def diagnostic_plots(df, variable, target):
    plt.figure(figsize=(20, 7))

    # Histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df[variable], kde=True, color='r')
    plt.title('Histogram')

    # Scatterplot
    plt.subplot(1, 4, 2)
    plt.scatter(df[variable], df[target], color='g')
    plt.title('Scatterplot')

    # Boxplot
    plt.subplot(1, 4, 3)
    sns.boxplot(y=df[variable], color='b')
    plt.title('Boxplot')

    # Barplot (skip if variable numeric)
    if df[variable].dtype == 'object' or df[variable].nunique() < 10:
        plt.subplot(1, 4, 4)
        sns.barplot(x=variable, y=target, data=df)
        plt.title('Barplot')

    plt.show()

# Generate diagnostic plots
for col in df.columns:
    if col != 'Survived':
        diagnostic_plots(df, col, 'Survived')

# Modeling
X = df.drop(columns=['Survived'])
y = df['Survived'].values  # 1D array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

# Scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN error and accuracy
K = 40
error = []
accuracy = []
for i in range(1, K + 1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(1 - metrics.accuracy_score(y_test, y_pred))
    accuracy.append(metrics.accuracy_score(y_test, y_pred))

# Plot error and accuracy
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(range(1, K+1), error, 'r-o')
plt.xlabel('K')
plt.ylabel('Error')
plt.title('Error vs K')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, K+1), accuracy, 'r-o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K')
plt.grid()
plt.show()

# Best K using GridSearchCV
param_grid = {
    'n_neighbors': [7, 9, 11, 13, 15, 25, 34],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_kn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_kn.fit(X_train, y_train)

best_knn = grid_kn.best_estimator_
y_pred_best = best_knn.predict(X_test)
accuracy_best = metrics.accuracy_score(y_test, y_pred_best)
print("Best KNN accuracy:", accuracy_best)
print("Best parameters:", grid_kn.best_params_)

# Classification report & confusion matrix
print(metrics.classification_report(y_test, y_pred_best, digits=4))
cm = metrics.confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
