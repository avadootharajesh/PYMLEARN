# ML_Algorithms.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------------------------
# Classification Example
# --------------------------
print("\n--- Classification: Breast Cancer Dataset ---")
data = load_breast_cancer()
X_clf = pd.DataFrame(data.data, columns=data.feature_names)
y_clf = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

for name, model in clf_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------
# Regression Example
# --------------------------
print("\n--- Regression: Diabetes Dataset ---")
data = load_diabetes()
X_reg = pd.DataFrame(data.data, columns=data.feature_names)
y_reg = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor()
}

for name, model in reg_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} RMSE: {rmse:.2f}, R²: {r2:.3f}")

# --------------------------
# Clustering Example
# --------------------------
print("\n--- Clustering: KMeans on Synthetic Data ---")
X_cluster, _ = make_blobs(n_samples=300, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)
score = silhouette_score(X_cluster, clusters)
print(f"Silhouette Score (KMeans): {score:.3f}")

# Visualize Clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title("KMeans Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()
