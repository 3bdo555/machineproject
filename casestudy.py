import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Inspect the dataset
df = pd.read_csv("titanic.csv")
print(df.head()) 
print(df.info())
print(df.isnull().sum())

# Drop irrelevant columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# 3. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 4. Feature Engineering

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Extract Title from Name (if Name exists)
if "Name" in df.columns:
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\s*\.", expand=False)
else:
    df["Title"] = "Unknown"

# Simplify rare titles
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# 5. Encode categorical variables
label_cols = ["Sex", "Embarked", "Title"]

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 6. Scale numerical features
num_cols = ["Age", "Fare", "FamilySize"]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 7. Final check
print("✅ Preprocessing complete!")
print(df.to_string()) # prints the whole dataset
