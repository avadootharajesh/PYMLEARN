# Interactive_Data_Dashboard_for_Sales_Analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_sample_data():
    # Generate sample sales data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'Date': np.random.choice(dates, 1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Furniture'], 1000),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
        'Units Sold': np.random.randint(1, 20, 1000),
        'Unit Price': np.random.uniform(10, 100, 1000)
    }
    df = pd.DataFrame(data)
    df['Sales'] = df['Units Sold'] * df['Unit Price']
    return df

def main():
    st.title("📊 Interactive Sales Analytics Dashboard")

    df = load_sample_data()

    # Sidebar filters
    st.sidebar.header("Filter Options")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    regions = st.sidebar.multiselect("Select Region(s)", options=df['Region'].unique(), default=df['Region'].unique())
    categories = st.sidebar.multiselect("Select Category(ies)", options=df['Category'].unique(), default=df['Category'].unique())

    # Filter data based on selections
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Region'].isin(regions)) &
        (df['Category'].isin(categories))
    ]

    # KPIs
    total_sales = filtered_df['Sales'].sum()
    total_units = filtered_df['Units Sold'].sum()
    avg_unit_price = filtered_df['Unit Price'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales ($)", f"{total_sales:,.2f}")
    col2.metric("Total Units Sold", f"{total_units}")
    col3.metric("Average Unit Price ($)", f"{avg_unit_price:.2f}")

    # Sales Over Time
    sales_over_time = filtered_df.groupby('Date').agg({'Sales':'sum'}).reset_index()
    fig1 = px.line(sales_over_time, x='Date', y='Sales', title="Sales Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # Sales by Region
    sales_by_region = filtered_df.groupby('Region').agg({'Sales':'sum'}).reset_index()
    fig2 = px.bar(sales_by_region, x='Region', y='Sales', title="Sales by Region", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Sales by Category
    sales_by_category = filtered_df.groupby('Category').agg({'Sales':'sum'}).reset_index()
    fig3 = px.pie(sales_by_category, values='Sales', names='Category', title="Sales Distribution by Category")
    st.plotly_chart(fig3, use_container_width=True)

    # Top Products
    top_products = filtered_df.groupby('Product').agg({'Sales':'sum'}).reset_index().sort_values(by='Sales', ascending=False)
    st.subheader("Top Products by Sales")
    st.dataframe(top_products)

if __name__ == "__main__":
    main()

# streamlit run interactive_sales_dashboard.py
