import pandas as pd

TARGET_CATEGORY = 'agro_industria_e_comercio'
DATE_COLUMN = 'order_purchase_timestamp'


def create_weekly_time_series():
    """
    Loads master data and aggregates sales by WEEK instead of day.
    """
    print("⏳ Creating WEEKLY time series data...")

    try:
        df = pd.read_csv('data/master_agri_data.csv')
    except FileNotFoundError:
        print("Error: master_agri_data.csv not found.")
        return

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df_ts = df[df['product_category_name'] == TARGET_CATEGORY].copy()
    
    if df_ts.empty:
        print(f"Error: No orders found for the category '{TARGET_CATEGORY}'.")
        return

    # 1. Aggregate: Count the number of orders per day first
    daily_sales = df_ts.groupby(df_ts[DATE_COLUMN].dt.date)['order_id'].count()
    ts_df = daily_sales.rename('sales_volume').to_frame()
    ts_df.index = pd.to_datetime(ts_df.index)

    # 2. KEY CHANGE: Resample to WEEKLY frequency ('W') and sum the sales
    weekly_sales = ts_df.resample('W').sum()

    # 3. Rename and save the new weekly data
    weekly_sales.columns = ['sales_volume']
    weekly_sales.to_csv('data/ts_sales_weekly.csv', header=True)
    
    print(f"✅ WEEKLY Time Series created. Saved to data/ts_sales_weekly.csv")


if __name__ == '__main__':
    create_weekly_time_series()