import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_elbow_method():
    """
    Runs K-Means for K=2 to 10 and calculates the Inertia (WCSS) score for each K.
    """
    print("⏳ Running Elbow Method: Calculating Inertia for K=2 to 10...")

    try:
        # Load the scaled data
        rfm_scaled_df = pd.read_csv('data/rfm_scaled_data.csv')
    except FileNotFoundError:
        print("Error: data/rfm_scaled_data.csv not found.")
        return

    # Select only the scaled features for clustering
    X = rfm_scaled_df[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    
    # Define the range of K values to test
    k_range = range(2, 11)
    inertia_scores = []

    for k in k_range:
        # Run K-Means with the specified K
        kmeans_model = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10 # n_init=10 is the default and standard practice
        )
        kmeans_model.fit(X)
        
        # Collect the inertia score (WCSS)
        inertia_scores.append(kmeans_model.inertia_)
        print(f"  K={k} Inertia calculated.")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_scores, marker='o', linestyle='--')
    plt.title('Elbow Method to Determine Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    plt.show() # Displays the plot

    print("\n✅ Elbow method complete. Analyzing plot for optimal K.")
    
    # Print the scores for analysis
    print("\n--- Inertia Scores by K ---")
    for k, score in zip(k_range, inertia_scores):
        print(f"K={k}: {score:.2f}")


if __name__ == '__main__':
    run_elbow_method()