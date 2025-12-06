import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ===============================================
# DATA LOADING AND SPLITTING
# ===============================================

print("⏳ Loading and preparing data...")
df = pd.read_csv('data/master_agri_data.csv')

# Define features and target
features = ['price', 'freight_value', 'product_weight_g', 'product_category_name', 'seller_state']
target = 'is_late'

df_model = df[features + [target]].copy()

# Cleaning: Drop any rows with missing values
df_model = df_model.dropna()
print(f"Total model records: {len(df_model)} rows")

# Define X (Inputs) and y (Output)
X = df_model[features]
y = df_model[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} rows")
print(f"Testing set:  {len(X_test)} rows")


# ===============================================
# PREPROCESSING AND MODEL TRAINING
# ===============================================

print("🛠 Preprocessing data with One-Hot Encoder...")

# Define which columns are numerical vs. categorical (text)
numerical_features = ['price', 'freight_value', 'product_weight_g']
categorical_features = ['product_category_name', 'seller_state']

# Create the Preprocessor (One-Hot Encoder for text columns)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply the Preprocessor to the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 6. TRAIN THE MODEL 🧠
print("🧠 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)


# 7. EVALUATE THE MODEL 💯
print("💯 Grading model performance...")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Test Data: {accuracy:.2f}")


# 8. SAVE THE MODEL
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("💾 Model and Preprocessor saved as model.pkl and preprocessor.pkl")