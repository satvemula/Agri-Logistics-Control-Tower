import pandas as pd
from datetime import datetime

def calculate_rfm():
    """
    Loads order data and calculates Recency, Frequency, and Monetary scores for each customer.
    """
    print("⏳ Starting RFM ETL: Calculating Recency, Frequency, and Monetary scores...")

    try:
        # Load the raw Olist order data
        # We need this raw file because the master file was focused on the 'agro' product.
        # We will use the raw customer and order data for a full customer view.
        orders_df = pd.read_csv('data/olist_orders_dataset.csv')
        items_df = pd.read_csv('data/olist_order_items_dataset.csv')
        payments_df = pd.read_csv('data/olist_order_payments_dataset.csv')
        customers_df = pd.read_csv('data/olist_customers_dataset.csv')
    except FileNotFoundError:
        print("Error: Required raw Olist data files not found in the 'data/' folder.")
        return

    # --- Data Cleaning and Preparation ---
    
    # 1. Merge all relevant data into a single transaction table
    transactions = orders_df.merge(items_df, on='order_id')
    transactions = transactions.merge(payments_df, on='order_id')
    transactions = transactions.merge(customers_df, on='customer_id')

    # Drop orders that were cancelled or unavailable
    transactions = transactions[transactions['order_status'].isin(['delivered', 'shipped'])]
    
    # Use the purchase date
    transactions['order_purchase_timestamp'] = pd.to_datetime(transactions['order_purchase_timestamp'])

    # --- RFM Calculation ---
    
    # Define a snapshot date (the day AFTER the last purchase in the dataset)
    snapshot_date = transactions['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    print(f"Snapshot date set to: {snapshot_date.date()}")

    # Calculate RFM scores
    rfm_df = transactions.groupby('customer_unique_id').agg(
        # R: Recency (Days since last order)
        Recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
        # F: Frequency (Total number of orders)
        Frequency=('order_id', 'nunique'),
        # M: Monetary (Total spend)
        Monetary=('payment_value', 'sum')
    )

    # Filter out customers with zero monetary value (cancelled payments, etc.)
    rfm_df = rfm_df[rfm_df['Monetary'] > 0]

    # Save the RFM table
    rfm_df.reset_index().to_csv('data/rfm_customer_data.csv', index=False)
    print(f"✅ RFM table created for {len(rfm_df)} unique customers. Saved to data/rfm_customer_data.csv")


if __name__ == '__main__':
    calculate_rfm()