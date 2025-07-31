# Time_Series.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Generate sample time series data: Monthly sales with trend + seasonality + noise
np.random.seed(42)
date_rng = pd.date_range(start='2010-01-01', periods=120, freq='M')
trend = np.linspace(10, 50, 120)
seasonality = 10 + 5 * np.sin(2 * np.pi * date_rng.month / 12)
noise = np.random.normal(0, 2, 120)
sales = trend + seasonality + noise
ts = pd.Series(sales, index=date_rng)

# Plot time series
ts.plot(title='Monthly Sales')
plt.show()

# Check stationarity with ADF test
adf_result = adfuller(ts)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] > 0.05:
    print("Series is non-stationary; consider differencing")

# Differencing to achieve stationarity
ts_diff = ts.diff().dropna()

# Fit ARIMA model (p,d,q) = (2,1,2)
model = ARIMA(ts, order=(2,1,2))
model_fit = model.fit()

print(model_fit.summary())

# Forecast next 12 periods
forecast = model_fit.forecast(steps=12)
print("Forecasted values:")
print(forecast)

# Plot forecast
plt.figure(figsize=(10,5))
plt.plot(ts, label='Historical')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.legend()
plt.show()
