# Real_World_Datasets.py
import pandas as pd
import seaborn as sns

# Load Titanic dataset (via seaborn)
titanic = sns.load_dataset('titanic')

# Basic info
print(titanic.info())

# Check missing values
print(titanic.isnull().sum())

# Preview data
print(titanic.head())

# Simple value counts for 'survived'
print(titanic['survived'].value_counts())

# Visualize survival by sex
import matplotlib.pyplot as plt
sns.barplot(x='sex', y='survived', data=titanic)
plt.show()
