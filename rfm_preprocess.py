import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_rfm():
    """
    Loads RFM data, applies log transformation to reduce skewness, 
    and then applies standard scaling for K-Means clustering.
    """
    print("⏳ Starting RFM Preprocessing: Log Transform and Scaling...")

    try:
        rfm_df = pd.read_csv('data/rfm_customer_data.csv')
    except FileNotFoundError:
        print("Error: data/rfm_customer_data.csv not found. Run the RFM ETL first.")
        return

    # Select the RFM features
    rfm_data = rfm_df[['Recency', 'Frequency', 'Monetary']]

    # 1. Log Transformation (np.log is log base e)
    # We use log(1+x) because Frequency has many '1's, and log(1) = 0.
    rfm_log = np.log1p(rfm_data)
    print("Log transformation complete.")

    # 2. Standard Scaling
    scaler = StandardScaler()
    rfm_scaled_array = scaler.fit_transform(rfm_log)

    # Convert the scaled array back to a DataFrame
    rfm_scaled_df = pd.DataFrame(
        rfm_scaled_array, 
        columns=['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']
    )
    
    # Add the customer_unique_id back for later analysis
    rfm_scaled_df['customer_unique_id'] = rfm_df['customer_unique_id']
    
    # Save the final scaled data
    rfm_scaled_df.to_csv('data/rfm_scaled_data.csv', index=False)
    print(f"✅ Scaled RFM data saved to data/rfm_scaled_data.csv")
    
    # Print the first 5 rows of the scaled data
    print("\n--- First 5 Rows of SCALED RFM Data (Mean=0, StdDev=1) ---")
    print(rfm_scaled_df.head())


if __name__ == '__main__':
    preprocess_rfm()