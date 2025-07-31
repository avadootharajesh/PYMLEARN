# Real_Estate_Price_Predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Using Boston Housing Dataset as a sample (replace with your own dataset)
    from sklearn.datasets import load_boston
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    return df

def preprocess_data(df):
    # For Boston dataset, no categorical vars; if your data has, encode them here
    return df

def feature_target_split(df):
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Plot true vs predicted
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=preds)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    df = preprocess_data(df)
    X, y = feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
