# Data_Cleaning.py
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward', 'Alice', None],
    'Age': [25, np.nan, 35, 45, 29, 25, 32],
    'Salary': [50000, 60000, None, 80000, 70000, 50000, 62000],
    'JoinDate': ['2021-01-01', '2020-07-15', 'Not Available', '2019-05-30', '2022-11-01', '2021-01-01', '2021-08-09'],
    'Department': ['HR', 'Finance', 'HR', 'IT', 'Finance', 'HR', 'Finance'],
    'Extra_Column': ['Remove', 'Remove', 'Remove', 'Remove', 'Remove', 'Remove', 'Remove']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Data:")
print(df)

# 1. Handling Missing Values
print("\n--- Handling Missing Values ---")
print("Missing values before:")
print(df.isnull().sum())

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Salary' with mean
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Fill missing 'Name' with 'Unknown'
df['Name'].fillna('Unknown', inplace=True)

print("Missing values after:")
print(df.isnull().sum())

# 2. Removing Duplicates
print("\n--- Removing Duplicates ---")
print("Duplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# 3. Fixing Data Types
print("\n--- Fixing Data Types ---")
# Convert JoinDate to datetime
df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce')
print(df.dtypes)

# 4. Handling Outliers (using IQR)
print("\n--- Handling Outliers ---")
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
print("Data after removing outliers based on Age:")
print(df)

# 5. String Cleaning
print("\n--- String Cleaning ---")
df['Name'] = df['Name'].str.strip().str.lower()
print(df['Name'])

# 6. Encoding Categorical Variables
print("\n--- Encoding Categorical Variables ---")
df = pd.get_dummies(df, columns=['Department'], drop_first=True)
print(df.head())

# 7. Renaming Columns
print("\n--- Renaming Columns ---")
df.rename(columns={'JoinDate': 'Joining_Date'}, inplace=True)
print(df.columns)

# 8. Dropping Irrelevant Features
print("\n--- Dropping Irrelevant Features ---")
df.drop(columns=['Extra_Column'], inplace=True)
print(df.head())

# 9. Saving Cleaned Data
print("\n--- Saving Cleaned Data ---")
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved to 'cleaned_data.csv'")