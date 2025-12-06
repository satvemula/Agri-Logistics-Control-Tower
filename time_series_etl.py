import pandas as pd

# Define the target category and date column
TARGET_CATEGORY = 'agro_industria_e_comercio'
DATE_COLUMN = 'order_purchase_timestamp'


def create_time_series_data():
    """
    Loads master data, filters for the target category, and aggregates sales by day.
    """
    print("⏳ Loading master data and preparing time series...")

    # Load the master data file
    try:
        df = pd.read_csv('data/master_agri_data.csv')
    except FileNotFoundError:
        print("Error: master_agri_data.csv not found. Ensure it is in the 'data/' folder.")
        return

    # 1. Convert the date column to datetime objects
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # 2. Filter for the target product category
    df_ts = df[df['product_category_name'] == TARGET_CATEGORY].copy()
    
    if df_ts.empty:
        print(f"Error: No orders found for the category '{TARGET_CATEGORY}'. Cannot create time series.")
        return

    # 3. Aggregate: Count the number of orders per day
    daily_sales = df_ts.groupby(df_ts[DATE_COLUMN].dt.date)['order_id'].count()
    
    # Convert the Series to a DataFrame
    ts_df = daily_sales.rename('sales_volume').to_frame()
    ts_df.index = pd.to_datetime(ts_df.index)

    # 4. Re-index to ensure a continuous time series
    # Find the earliest and latest dates in the sales data
    start_date = ts_df.index.min()
    end_date = ts_df.index.max()
    
    # Create a continuous date range and align the sales data
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    ts_df = ts_df.reindex(full_range, fill_value=0)

    print(f"✅ Time Series created from {start_date.date()} to {end_date.date()}.")
    print(f"Total days: {len(ts_df)}. Saved to data/ts_sales.csv")

    # Save the final time series data
    ts_df.to_csv('data/ts_sales.csv', header=True)


if __name__ == '__main__':
    create_time_series_data()