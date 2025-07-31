# Smart_Expense_Tracker_with_Anomaly_Detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns

def generate_sample_expenses(n=200):
    np.random.seed(42)
    categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Health', 'Shopping']
    
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=n, freq='D'),
        'Category': np.random.choice(categories, size=n),
        'Amount': np.random.normal(loc=50, scale=20, size=n).round(2),
    }
    df = pd.DataFrame(data)
    
    # Add some anomalies (large unexpected expenses)
    anomalies_idx = np.random.choice(n, size=5, replace=False)
    df.loc[anomalies_idx, 'Amount'] *= 5  # Inflate anomaly amounts
    
    # Ensure no negative amounts
    df['Amount'] = df['Amount'].apply(lambda x: max(x, 1))
    return df

def detect_anomalies(df):
    # Use Amount and encode category for anomaly detection
    df_encoded = pd.get_dummies(df['Category'])
    X = pd.concat([df[['Amount']], df_encoded], axis=1)

    model = IsolationForest(contamination=0.025, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    # anomaly == -1 means anomaly
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    return df, model

def plot_expenses(df):
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, x='Date', y='Amount', hue='anomaly', palette={0:'blue',1:'red'})
    plt.title("Expense Amounts Over Time with Anomalies Highlighted")
    plt.xlabel("Date")
    plt.ylabel("Amount ($)")
    plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'])
    plt.show()

def main():
    print("Generating sample expense data...")
    expenses_df = generate_sample_expenses()
    print(expenses_df.head())

    print("\nDetecting anomalies...")
    expenses_df, model = detect_anomalies(expenses_df)
    print(expenses_df[expenses_df['anomaly'] == 1])

    print("\nPlotting expenses with anomalies...")
    plot_expenses(expenses_df)

if __name__ == "__main__":
    main()
