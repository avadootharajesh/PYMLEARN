# Employee_Attrition_Prediction_System.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_sample_data():
    # Sample synthetic dataset for demonstration
    data = {
        'Age': np.random.randint(22, 60, 300),
        'Department': np.random.choice(['Sales', 'R&D', 'HR'], 300),
        'JobRole': np.random.choice(['Manager', 'Engineer', 'Sales Executive'], 300),
        'MonthlyIncome': np.random.randint(3000, 15000, 300),
        'JobSatisfaction': np.random.randint(1, 5, 300),
        'YearsAtCompany': np.random.randint(0, 20, 300),
        'WorkLifeBalance': np.random.randint(1, 5, 300),
        'Attrition': np.random.choice(['Yes', 'No'], 300, p=[0.2, 0.8])
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Encode categorical columns
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

    for col in ['Department', 'JobRole']:
        df[col] = le.fit_transform(df[col])

    # Feature scaling
    scaler = StandardScaler()
    features = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'WorkLifeBalance']
    df[features] = scaler.fit_transform(df[features])

    X = df[features]
    y = df['Attrition']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    roc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_score:.2f}")

def main():
    print("Loading data...")
    df = load_sample_data()
    print(df.head())

    print("\nPreprocessing data...")
    X, y = preprocess_data(df)

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
