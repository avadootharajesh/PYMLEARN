# Problem_Solving.py
# problem_solving_template.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def understand_problem():
    print("Problem: Predict if a customer will churn based on behavior data.")
    print("Success metric: Accuracy above 80%.")

def load_and_explore_data(path):
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print("Sample data:")
    print(df.head())
    print("Summary statistics:")
    print(df.describe())
    print("Missing values per column:")
    print(df.isnull().sum())
    return df

def clean_and_preprocess(df):
    df = df.dropna()  # Simple approach, customize as needed
    # Example encoding categorical variable if any
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    return df

def feature_engineering(df):
    # Example: create new feature 'total_spent' if relevant columns exist
    if 'monthly_charges' in df.columns and 'tenure' in df.columns:
        df['total_spent'] = df['monthly_charges'] * df['tenure']
    return df

def train_and_evaluate(df, target_col='churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))

def main():
    understand_problem()
    df = load_and_explore_data('customer_churn.csv')
    df = clean_and_preprocess(df)
    df = feature_engineering(df)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
