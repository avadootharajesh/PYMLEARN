# Feature_Engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Load Sample Dataset
# --------------------------
# Simulated Titanic-like dataset
data = {
    'Name': ['John Doe', 'Emily Smith', 'Alice Brown', 'Mike Johnson', 'Sarah Lee'],
    'Age': [22, 38, 26, 45, 29],
    'Fare': [7.25, 71.83, 8.05, 8.05, 13.00],
    'Sex': ['male', 'female', 'female', 'male', 'female'],
    'Pclass': [3, 1, 3, 3, 3],
    'Cabin': ['C85', np.nan, np.nan, np.nan, 'E46'],
    'Embarked': ['S', 'C', 'S', 'S', 'C'],
    'Survived': [0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

print("\n--- Original Data ---")
print(df)

# --------------------------
# Feature: Title Extraction from Name
# --------------------------
df['Title'] = df['Name'].apply(lambda x: x.split(' ')[0])
print("\n--- Feature Created: Title ---")
print(df[['Name', 'Title']])

# --------------------------
# Categorical Encoding
# --------------------------
# Label Encoding for Sex
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# --------------------------
# Handling Missing Values
# --------------------------
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Cabin_type'] = df['Cabin'].str[0]  # Extract deck letter

# --------------------------
# Binning Age
# --------------------------
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Teen', 'Young', 'Adult', 'Senior'])

# --------------------------
# Scaling Numerical Features
# --------------------------
scaler = StandardScaler()
df[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(df[['Age', 'Fare']])

# --------------------------
# Feature Selection
# --------------------------
# Define input features and target
X = df[['Pclass', 'Sex_encoded', 'Fare_scaled', 'Age_scaled']]
y = df['Survived']

selector = SelectKBest(score_func=f_classif, k='all')
fit = selector.fit(X, y)
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': fit.scores_})
print("\n--- Feature Selection Scores ---")
print(feature_scores)

# --------------------------
# Dimensionality Reduction with PCA
# --------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\n--- PCA Explained Variance Ratio ---")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("PCA - Feature Engineering Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
