import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ===============================================
# PREPROCESSOR SETUP
# ===============================================

numerical_features = ['price', 'freight_value', 'product_weight_g']
categorical_features = ['product_category_name', 'seller_state']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)


# ===============================================
# DATA LOADING AND SPLITTING
# ===============================================

print("⏳ Loading and preparing data...")
df = pd.read_csv('data/master_agri_data.csv')

features = ['price', 'freight_value', 'product_weight_g', 'product_category_name', 'seller_state']
target = 'is_late'

df_model = df[features + [target]].copy().dropna()
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} rows")
print(f"Testing set:  {len(X_test)} rows")


# --- DATA PROCESSING ---
print("🛠 Processing data with One-Hot Encoder...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ===============================================
# FINAL OPTIMIZATION: GRID SEARCH FOR F1-SCORE
# ===============================================

# 1. Define the F1-Score as the target metric for optimization
#   We use average='binary' to focus the score only on the positive class (1/Late)
f1_scorer = make_scorer(f1_score, average='binary')

# 2. Define the parameter grid (values of scale_pos_weight to test)
#    We will test weights from 10 to 30. Our previous best was 18.
param_grid = {
    'scale_pos_weight': [10, 15, 20, 25, 30]
}

# 3. Setup the XGBoost model for the Grid Search
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 4. Perform Grid Search
print("\n🔎 Running Grid Search to find best F1-Score...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=3, # Use 3-fold cross-validation
    verbose=1
)

grid_search.fit(X_train_processed, y_train)

# 5. Extract the Best Model and Parameters
best_model = grid_search.best_estimator_
best_weight = grid_search.best_params_['scale_pos_weight']

print("\n--- GRID SEARCH RESULTS ---")
print(f"✅ Best F1-Score achieved: {grid_search.best_score_:.4f}")
print(f"⚖️ Best scale_pos_weight found: {best_weight}")


# ===============================================
# FINAL EVALUATION OF OPTIMIZED MODEL
# ===============================================

# 6. Evaluate the best model on the original test set
print("💯 Grading BEST Model Performance on original test data...")
y_pred = best_model.predict(X_test_processed)

print("\n--- FINAL CLASSIFICATION REPORT (MAX F1) ---")
print("Target: 0 = On Time | 1 = Late")
print(classification_report(y_test, y_pred, target_names=['0 (On Time)', '1 (Late)']))


# 7. SAVE THE OPTIMIZED MODEL
joblib.dump(best_model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print(f"💾 Optimized Model (Weight: {best_weight}) and Preprocessor saved.")