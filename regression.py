import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


data = pd.read_csv("titanic_cleaned.csv")
y = data['Fare']

print("REGRESSION MODELS")
print(f"Target: Fare (mean=${y.mean():.2f}, std=${y.std():.2f})\n")


#  Simple Linear Regression

print("1. Simple Linear Regression (Age → Fare)")

X_simple = data[['Age']]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_simple = model_simple.predict(X_test_s)

r2_simple, rmse_simple = evaluate_model(y_test_s, y_pred_simple)

print(f"R² Score: {r2_simple:.4f}")
print(f"RMSE: ${rmse_simple:.2f}")
print(f"Coefficient (Age): {model_simple.coef_[0]:.3f}")
print(f"Intercept: ${model_simple.intercept_:.2f}")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test_s, y_pred_simple, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Simple Linear Regression: Actual vs Predicted Fare')
plt.show()


#  Multiple Linear Regression

print("2. Multiple Linear Regression (All features → Fare)")

features_multi = ['Age', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked']
X_multi = data[features_multi]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Sex', 'Embarked']),
        ('num', 'passthrough', ['Age', 'Pclass', 'SibSp', 'Parch'])
    ]
)

model_multi = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_multi.fit(X_train_m, y_train_m)
y_pred_multi = model_multi.predict(X_test_m)

r2_multi, rmse_multi = evaluate_model(y_test_m, y_pred_multi)

print(f"R² Score: {r2_multi:.4f}")
print(f"RMSE: ${rmse_multi:.2f}")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test_m, y_pred_multi, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Multiple Linear Regression: Actual vs Predicted Fare')
plt.show()


#  Polynomial Regression
print("3. Polynomial Regression (Age → Fare)")

X_poly = data[['Age']]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train_p, y_train_p)
y_pred_poly = poly_model.predict(X_test_p)

r2_poly, rmse_poly = evaluate_model(y_test_p, y_pred_poly)

print(f"R² Score: {r2_poly:.4f}")
print(f"RMSE: ${rmse_poly:.2f}")

# Plot
X_plot = np.linspace(X_poly.min(), X_poly.max(), 300).reshape(-1, 1)
y_plot = poly_model.predict(X_plot)

plt.figure(figsize=(8,6))
plt.scatter(X_test_p, y_test_p, color='purple', alpha=0.6, label='Actual')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Degree 2 Fit')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Polynomial Regression (Degree 2): Age vs Fare')
plt.legend()
plt.show()

# Comparison
comparison = pd.DataFrame({
    'Model': [
        'Simple Linear (Age)',
        'Multiple Linear (All features)',
        'Polynomial (Degree 2)'
    ],
    'R² Score': [
        round(r2_simple, 4),
        round(r2_multi, 4),
        round(r2_poly, 4)
    ],
    'RMSE ($)': [
        round(rmse_simple, 2),
        round(rmse_multi, 2),
        round(rmse_poly, 2)
    ]
})

print("MODEL COMPARISON")
print(comparison.to_string(index=False))
