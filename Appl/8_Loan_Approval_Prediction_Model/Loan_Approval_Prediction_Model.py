# Loan_Approval_Prediction_Model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    # Using sample loan dataset from Kaggle (replace URL if needed)
    url = "https://raw.githubusercontent.com/selva86/datasets/master/LoanApproval.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Fill missing values with mode for categorical, median for numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def feature_target_split(df):
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    df = load_data()
    df = preprocess_data(df)
    X, y = feature_target_split(df)
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
