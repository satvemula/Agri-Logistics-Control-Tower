# ==============================================================================
# ETL PIPELINE (Extract, Transform, Load) - AGRI MODE
# ==============================================================================
import pandas as pd
import os

DATA_DIR = 'data' 

def load_data():
    print("⏳ Loading raw CSV files...")
    orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
    items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(DATA_DIR, 'olist_sellers_dataset.csv'))
    geolocation = pd.read_csv(os.path.join(DATA_DIR, 'olist_geolocation_dataset.csv'))
    return orders, items, products, sellers, geolocation

def merge_data(orders, items, products, sellers, geolocation):
    print("🔗 Merging tables...")
    
    # 1. Base Merges
    df = pd.merge(orders, items, on='order_id', how='inner')
    df = pd.merge(df, products, on='product_id', how='inner')
    df = pd.merge(df, sellers, on='seller_id', how='inner')
    
    # 2. Geolocation Merge
    geo_clean = geolocation.groupby('geolocation_zip_code_prefix').first().reset_index()
    df = pd.merge(df, geo_clean, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df = df.rename(columns={'geolocation_lat': 'lat', 'geolocation_lng': 'lon'})
    
    # 3. STRICT AGRI-FILTER (Portuguese Names)
    # We only want these specific categories
    ag_categories = [
        'agro_industria_e_comercio', 
        'ferramentas_jardim',           # Garden Tools
        'construcao_ferramentas_jardim',
        'alimentos',                    # Food
        'alimentos_bebidas',
        'climatizacao'                  # Air Conditioning
    ]
    
    # The Filtering Step
    df_ag = df[df['product_category_name'].isin(ag_categories)].copy()
    
    print(f"✅ Filtered down to {len(df_ag)} Agri-Logistics records.")
    return df_ag

def feature_engineering(df):
    print("🛠 Engineering features...")
    cols_to_fix = ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']
    for col in cols_to_fix:
        df[col] = pd.to_datetime(df[col])
        
    df['actual_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_late'] = df['delay_days'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def save_master(df):
    output_path = os.path.join(DATA_DIR, 'master_agri_data.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Success! Saved {len(df)} rows to: {output_path}")

if __name__ == "__main__":
    orders, items, products, sellers, geo = load_data()
    df_merged = merge_data(orders, items, products, sellers, geo)
    df_clean = feature_engineering(df_merged)
    save_master(df_clean)