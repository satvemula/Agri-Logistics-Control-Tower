import pandas as pd

try:
    df = pd.read_csv('data/master_agri_data.csv')
    
    # Print all unique values in the category column
    unique_categories = df['product_category_name'].dropna().unique()
    
    print("\n--- Unique Product Categories Found ---")
    for category in unique_categories:
        print(category)
        
except FileNotFoundError:
    print("Error: master_agri_data.csv not found.")
except KeyError:
    print("Error: 'product_category_name' column not found.")