import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("titanic_cleaned.csv")
print("Loaded cleaned data:")
print(df.head())


features = df.select_dtypes(include=[np.number]).drop(columns=['Survived'])
print("\nFeatures used for K-Means:")
print(features.head())


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
print("\nScaled features (first 5 rows):")
print(scaled_df.head())

# 3) Elbow method
sse = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel('K')
plt.ylabel('Sum of Squared Error (Inertia)')
plt.title('Elbow Plot for K-Means (Titanic)')
plt.show()


k_opt = 3
km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(scaled_df)

print("\nCluster assignments (first 5 rows):")
print(df[['Survived', 'Age', 'Fare', 'cluster']].head())


centers_scaled = km.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=features.columns)
print("\nCluster centers (original scale):")
print(centers_df)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Age', y='Fare',
    hue='cluster',
    data=df,
    palette='viridis',
    alpha=0.7
)
plt.title('K-Means Clusters (Age vs Fare)')
plt.legend(title='Cluster')
plt.show()

print("\nSurvival rate by cluster (%):")
print((df.groupby('cluster', observed=False)['Survived'].mean() * 100).round(2))
