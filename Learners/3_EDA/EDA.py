# EDA.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# ---------------------------
# Load Sample Dataset
# ---------------------------
# Sample Titanic-like dataset
data = {
    'PassengerId': range(1, 11),
    'Name': ['John', 'Emily', 'Alex', 'Kate', 'Mike', 'Anna', 'Tom', 'Lucy', 'Sam', 'Jane'],
    'Age': [22, 38, 26, 35, np.nan, 28, 30, 40, np.nan, 19],
    'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'Fare': [7.25, 71.28, 8.05, 53.10, 8.46, 13.00, 8.05, 27.72, 8.46, 7.88],
    'Survived': [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    'Pclass': [3, 1, 3, 1, 3, 3, 3, 2, 3, 3],
    'Cabin': [None, 'C85', None, 'C123', None, None, None, 'E46', None, None]
}

df = pd.DataFrame(data)

# ---------------------------
# 1. Dataset Overview
# ---------------------------
print("\n--- Dataset Info ---")
print(df.info())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample Data:\n", df.head())

# ---------------------------
# 2. Summary Statistics
# ---------------------------
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# ---------------------------
# 3. Missing Values
# ---------------------------
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Visualize missing data
msno.matrix(df)
plt.title("Missing Data Matrix")
plt.show()

# ---------------------------
# 4. Univariate Analysis
# ---------------------------
print("\n--- Univariate Analysis ---")
# Numerical distributions
df['Age'].hist(bins=10)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x='Age', data=df)
plt.title('Boxplot of Age')
plt.show()

# Categorical distributions
print("\nSex Value Counts:\n", df['Sex'].value_counts())
sns.countplot(x='Sex', data=df)
plt.title('Gender Count')
plt.show()

# ---------------------------
# 5. Bivariate Analysis
# ---------------------------
print("\n--- Bivariate Analysis ---")
# Survival rate by sex
print(df.groupby('Sex')['Survived'].mean())

sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# Age vs Fare
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare (colored by Survival)')
plt.show()

# ---------------------------
# 6. Correlation Analysis
# ---------------------------
print("\n--- Correlation Matrix ---")
correlation = df[['Age', 'Fare', 'Survived', 'Pclass']].corr()
print(correlation)

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ---------------------------
# 7. Outlier Detection
# ---------------------------
print("\n--- Outlier Detection ---")
sns.boxplot(x='Fare')
plt.title('Boxplot of Fare')
plt.show()

# ---------------------------
# 8. Target Variable Analysis
# ---------------------------
print("\n--- Target Variable Distribution ---")
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# ---------------------------
# 9. Feature Selection
# ---------------------------
print("\n--- Feature Selection ---")
print(df.columns.tolist())