# Customer_Churn_Prediction_System.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Using Telco Customer Churn dataset from Kaggle (sample simulation)
    # You can replace with your own dataset path
    url = "https://raw.githubusercontent.com/blastchar/telco-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Drop customerID (unique identifier)
    df = df.drop(columns=['customerID'])
    
    # Convert 'TotalCharges' to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode target variable 'Churn'
    df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
    
    # Binary encode 'Yes'/'No' columns
    binary_cols = [col for col in df.columns if df[col].dtype == 'object' and set(df[col].unique()) == {'Yes','No'}]
    for col in binary_cols:
        df[col] = df[col].map({'No':0, 'Yes':1})
    
    # Encode remaining categorical columns with LabelEncoder
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

def feature_target_split(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print("Classification Report:\n", classification_report(y_test, preds))
    print("ROC-AUC Score:", roc_auc_score(y_test, proba))

    # Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_test.columns

    plt.figure(figsize=(12,6))
    sns.barplot(x=importances[indices], y=features[indices])
    plt.title('Feature Importances')
    plt.show()

def main():
    df = load_data()
    print(f"Original Data Shape: {df.shape}")
    
    df = preprocess_data(df)
    print(f"Processed Data Shape: {df.shape}")

    X, y = feature_target_split(df)

    # Split data (stratify to keep churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
