# Stock_Market_Trend_Analyzer.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data(ticker='AAPL', period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    return data

def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    return df

def plot_data(df, ticker):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['SMA_20'], label='20-Day SMA', color='red')
    plt.plot(df['SMA_50'], label='50-Day SMA', color='green')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

def detect_trends(df):
    # Simple trend signals based on SMA crossover
    df['Signal'] = 0
    df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1  # Bullish
    df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1  # Bearish
    return df

def main():
    ticker = 'AAPL'  # Apple Inc.
    df = fetch_data(ticker)
    df = add_technical_indicators(df)
    df = detect_trends(df)
    plot_data(df, ticker)

    print(df[['Close', 'SMA_20', 'SMA_50', 'Signal']].tail(10))

if __name__ == "__main__":
    main()
