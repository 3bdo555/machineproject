import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic_cleaned.csv")

data['Age'] = data['Age'].fillna(data['Age'].mean())

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'].fillna('S'))  

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  
y = data['Survived']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

results = pd.DataFrame({
    'Actual_Survived': y_test.values,
    'Predicted_Survived': y_pred
})

print(results.head(10))

results_full = X_test.copy()
results_full['Actual_Survived'] = y_test.values
results_full['Predicted_Survived'] = y_pred

print(results_full.head(10))

wrong_predictions = results_full[
    results_full['Actual_Survived'] != results_full['Predicted_Survived']
]

print(wrong_predictions.head(10))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
