import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

sns.set(style='whitegrid')
print('Libraries imported')

df = pd.read_csv('titanic_cleaned.csv')
print('Loaded:', df.shape)
df.head()

print('Info:')
print(df.info())

print('\nMissing values per column:')
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Pclass')
plt.show()

# Preprocessing
df_proc = df.copy()

X = df_proc.drop('Survived', axis=1)
y = df_proc['Survived']

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print('Numeric cols:', num_cols)
print('Categorical cols:', cat_cols)

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode()[0])

# One-hot encode categorical variables
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_cat = ohe.fit_transform(X[cat_cols])
cat_feature_names = ohe.get_feature_names_out(cat_cols)

X_num = X[num_cols].to_numpy()
X_prepared = np.hstack([X_num, X_cat])
feature_names = num_cols + list(cat_feature_names)

X = pd.DataFrame(X_prepared, columns=feature_names)
print('Prepared feature matrix shape:', X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print('Train size:', X_train.shape, 'Test size:', X_test.shape)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print('Trained Decision Tree')

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {acc:.4f}')

print('\nClassification report:')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

importances = clf.feature_importances_
fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
fi.head(15).plot(kind='bar')
plt.title('Top feature importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,12))
plot_tree(clf, feature_names=X.columns, class_names=['0','1'], filled=True, fontsize=8)
plt.title('Decision Tree')
plt.show()

import joblib
joblib.dump(clf, 'decision_tree_titanic_model.joblib')
print('Saved model to decision_tree_titanic_model.joblib')


