# COVID-19_Data_Analysis_and_Forecasting_Tool.py
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import requests
from io import StringIO

def load_covid_data(country='US'):
    # Download JHU time series confirmed cases CSV
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    
    # Filter country and aggregate by date
    country_data = data[data['Country/Region'] == country].drop(columns=['Province/State','Country/Region','Lat','Long'])
    country_data = country_data.sum().reset_index()
    country_data.columns = ['Date', 'Confirmed']
    country_data['Date'] = pd.to_datetime(country_data['Date'])
    return country_data

def plot_cases(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Confirmed'], label='Confirmed Cases')
    plt.title('COVID-19 Confirmed Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.legend()
    plt.grid(True)
    plt.show()

def forecast_cases(df, periods=30):
    # Prepare data for Prophet
    df_prophet = df.rename(columns={'Date':'ds', 'Confirmed':'y'})
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Plot forecast
    model.plot(forecast)
    plt.title('COVID-19 Confirmed Cases Forecast')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.show()
    
    return forecast

def main():
    country = 'US'
    print(f"Loading COVID-19 data for {country}...")
    df = load_covid_data(country)
    
    print("Plotting historical confirmed cases...")
    plot_cases(df)
    
    print("Forecasting future cases for next 30 days...")
    forecast = forecast_cases(df, periods=30)
    
    print("Forecast complete. Showing predicted values:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

if __name__ == "__main__":
    main()
