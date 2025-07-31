# Air_Quality_Index_Predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_sample_data(n=1000):
    # Generate synthetic air quality dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'PM2.5': np.random.uniform(5, 150, n),
        'PM10': np.random.uniform(10, 200, n),
        'CO': np.random.uniform(0.1, 10, n),
        'NO2': np.random.uniform(5, 100, n),
        'SO2': np.random.uniform(1, 50, n),
        'O3': np.random.uniform(10, 120, n),
        'Temperature': np.random.uniform(0, 40, n),
        'Humidity': np.random.uniform(10, 90, n),
        'WindSpeed': np.random.uniform(0, 15, n)
    })
    # AQI target: Simple weighted sum + noise
    data['AQI'] = (
        0.5 * data['PM2.5'] + 0.3 * data['PM10'] + 2 * data['CO'] +
        0.4 * data['NO2'] + 0.1 * data['SO2'] + 0.2 * data['O3'] +
        0.1 * data['Temperature'] - 0.05 * data['Humidity'] - 0.1 * data['WindSpeed'] +
        np.random.normal(0, 5, n)
    )
    return data

def train_model(df):
    X = df.drop('AQI', axis=1)
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Performance:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"R^2: {r2_score(y_test, y_pred):.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.grid(True)
    plt.show()

    return model

def predict_aqi(model, features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    print(f"Predicted AQI: {prediction:.2f}")
    return prediction

def main():
    data = generate_sample_data()
    model = train_model(data)

    # Example prediction
    sample_features = {
        'PM2.5': 55,
        'PM10': 80,
        'CO': 2.5,
        'NO2': 40,
        'SO2': 15,
        'O3': 30,
        'Temperature': 25,
        'Humidity': 50,
        'WindSpeed': 5
    }
    predict_aqi(model, sample_features)

if __name__ == "__main__":
    main()
