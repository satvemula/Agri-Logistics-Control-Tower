import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# --- 1. CONFIG AND SETUP ---

st.set_page_config(page_title="Agri-Logistics Control Tower", layout="wide")

# Paths
DATA_PATH = os.path.join('data', 'master_agri_data.csv')
MODEL_PATH = 'model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'

# Load Model and Preprocessor (The Brain)
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError:
    st.error("❌ Model files (model.pkl or preprocessor.pkl) not found. Run 'python training.py' first!")
    st.stop()


# --- 2. LOAD DATA ---

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ master_agri_data.csv not found.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()


# --- 3. SIDEBAR: PREDICTION INPUTS ---

st.sidebar.header("🤖 New Order Prediction")

# Input widgets for the model features
price = st.sidebar.number_input("Price (R$):", min_value=1.0, value=30.0)
freight_value = st.sidebar.number_input("Freight Cost (R$):", min_value=1.0, value=15.0)
weight = st.sidebar.number_input("Product Weight (g):", min_value=10.0, value=500.0)

# Get categorical options from the loaded data for the select boxes
all_categories = sorted(df['product_category_name'].dropna().unique())
all_states = sorted(df['seller_state'].dropna().unique())

category = st.sidebar.selectbox("Product Category:", options=all_categories, index=0)
state = st.sidebar.selectbox("Seller State:", options=all_states, index=2) # Default to SP (index 2)

# --- 4. PREDICTION LOGIC ---

# Create a DataFrame from the user inputs for the model
new_order = pd.DataFrame({
    'price': [price],
    'freight_value': [freight_value],
    'product_weight_g': [weight],
    'product_category_name': [category],
    'seller_state': [state]
})

# Preprocess the input data (One-Hot Encoding)
new_order_processed = preprocessor.transform(new_order)

# Make the prediction
prediction_proba = model.predict_proba(new_order_processed)[0]
prediction_class = model.predict(new_order_processed)[0]

# --- 5. MAIN DASHBOARD LAYOUT AND METRICS ---

st.title("🚜 Predictive Agri-Logistics Control Tower")

# --- Display Prediction Result ---
pred_col, acc_col = st.columns([2, 1])

with pred_col:
    st.subheader("Order Prediction Result")
    
    # Calculate probability of Lateness (class 1)
    late_proba = prediction_proba[1] 
    
    if prediction_class == 1:
        st.error(f"⚠️ Predicted: LATE DELIVERY")
        st.markdown(f"**Confidence:** {late_proba*100:.1f}% chance of being late.")
    else:
        st.success(f"✅ Predicted: ON TIME")
        st.markdown(f"**Confidence:** {(1 - late_proba)*100:.1f}% chance of being on time.")

with acc_col:
    st.metric("Model Accuracy", "93%")
    st.metric("Total Agri Revenue", f"${df['price'].sum():,.0f}")

st.divider()

# --- 6. INTERACTIVE MAP & ANALYSIS (from Week 1) ---

# Sidebar for historical data filtering (We repurpose the original state filter here)
st.sidebar.header("🎛️ Historical Filter")
selected_states = st.sidebar.multiselect(
    "Select Seller State(s) for Map:",
    options=all_states,
    default=all_states[:3] 
)

# Apply historical filter
if selected_states:
    df_filtered = df[df['seller_state'].isin(selected_states)]
else:
    df_filtered = df

col_map, col_chart = st.columns([2, 1])

with col_map:
    st.subheader("📍 Historical Seller Geography")
    map_data = df_filtered.dropna(subset=['lat', 'lon'])
    st.map(map_data)

with col_chart:
    st.subheader("⚠️ Worst Categories (Historical)")
    category_delay = df_filtered.groupby('product_category_name')['delay_days'].mean().reset_index()
    top_delay = category_delay.sort_values(by='delay_days', ascending=False).head(10)
    
    fig = px.bar(top_delay, x='delay_days', y='product_category_name', orientation='h',
                 title="Avg Delay by Category (Days)", color='delay_days', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)