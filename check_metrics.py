import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Reload the Data and Split (to recreate the exact test set)
print("⏳ Reloading data and calculating metrics...")
df = pd.read_csv('data/master_agri_data.csv').dropna()
features = ['price', 'freight_value', 'product_weight_g', 'product_category_name', 'seller_state']
X = df[features]
y = df['is_late']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Load the Saved Model and Preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# 3. Process the Test Data and Predict
X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

# 4. Print the Classification Report
print("\n--- FULL CLASSIFICATION REPORT ---")
print("Target: 0 = On Time | 1 = Late")
print(classification_report(y_test, y_pred, target_names=['0 (On Time)', '1 (Late)']))