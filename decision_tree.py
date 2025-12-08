import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_cleaned.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = ["Sex", "Embarked", "Pclass"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), cat_cols)],
    remainder="passthrough"
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf_gini.fit(X_train_enc, y_train)
pred_gini = clf_gini.predict(X_test_enc)
print("Gini Accuracy :", accuracy_score(y_test, pred_gini))

clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf_entropy.fit(X_train_enc, y_train)
pred_entropy = clf_entropy.predict(X_test_enc)
print("Entropy Accuracy :", accuracy_score(y_test, pred_entropy))

sns.heatmap(confusion_matrix(y_test, pred_gini), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Survived","Survived"], yticklabels=["Not Survived","Survived"])
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Gini")
plt.show()

print(classification_report(y_test, pred_gini, target_names=["Not Survived","Survived"]))

plt.figure(figsize=(20,10))
tree.plot_tree(clf_gini, feature_names=feature_names, class_names=["Not Survived","Survived"], filled=True, rounded=True)
plt.title("Decision Tree - Gini Index")
plt.show()




y_prob = clf_gini.predict_proba(X_test_enc)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree (Gini)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")