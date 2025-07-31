# Credit_Card_Fraud_Detection_System.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Features are mostly numeric; scale 'Amount' and 'Time'
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df = df.drop(['Time', 'Amount'], axis=1)
    return df

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_and_evaluate(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle imbalance with SMOTE on training data
    X_train_res, y_train_res = balance_data(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

def main():
    df = load_data()
    df = preprocess_data(df)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
