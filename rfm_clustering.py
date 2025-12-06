import pandas as pd
from sklearn.cluster import KMeans

def run_rfm_clustering():
    """
    Runs K-Means with K=4 and analyzes the resulting customer segments.
    """
    print("⏳ Running K-Means with K=4 and analyzing segments...")

    try:
        rfm_scaled_df = pd.read_csv('data/rfm_scaled_data.csv')
        rfm_original_df = pd.read_csv('data/rfm_customer_data.csv')
    except FileNotFoundError:
        print("Error: Required RFM data files not found.")
        return

    # Select only the scaled features for clustering
    X = rfm_scaled_df[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    
    # 1. Run K-Means with Optimal K=4
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X)
    
    # 2. Assign the cluster labels (0, 1, 2, 3) back to the original RFM data
    rfm_original_df['Cluster_Label'] = kmeans_model.labels_

    # 3. Analyze the Cluster Characteristics
    # Calculate the MEAN (average) of the original, unscaled RFM scores for each cluster.
    cluster_analysis = rfm_original_df.groupby('Cluster_Label').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'customer_unique_id': 'count' # Count how many customers are in each segment
    }).rename(columns={'customer_unique_id': 'Customer_Count'})
    
    # 4. Sort the results for easy interpretation (e.g., sort by Recency)
    cluster_analysis = cluster_analysis.sort_values(by='Recency', ascending=True)

    print("✅ K-Means clustering complete. Four customer segments found.")
    
    print("\n--- Final Customer Segment Analysis (K=4) ---")
    print(cluster_analysis.to_string())
    

if __name__ == '__main__':
    run_rfm_clustering()