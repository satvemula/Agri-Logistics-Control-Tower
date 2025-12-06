import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. PAGE SETUP
st.set_page_config(page_title="Agri-Logistics Control Tower", layout="wide")

# 2. LOAD DATA
@st.cache_data
def load_data():
    path = os.path.join('data', 'master_agri_data.csv')
    df = pd.read_csv(path)
    
    # Date conversion
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ master_agri_data.csv not found.")
    st.stop()

# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.sidebar.header("🎛️ Filters")

# Create a list of unique states, sorted alphabetically
all_states = sorted(df['seller_state'].unique())

# The Multiselect Widget
selected_states = st.sidebar.multiselect(
    "Select Seller State(s):",
    options=all_states,
    default=all_states[:3] # Default to first 3 states so the map isn't empty
)

# FILTER THE DATA BASED ON USER SELECTION
if selected_states:
    df_filtered = df[df['seller_state'].isin(selected_states)]
else:
    df_filtered = df # If nothing selected, show everything

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("🚜 Agri-Logistics Control Tower")
st.markdown(f"Showing data for: **{', '.join(selected_states) if selected_states else 'All States'}**")

# METRICS ROW
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", f"{len(df_filtered):,}")

with col2:
    # Revenue (Price column)
    total_rev = df_filtered['price'].sum()
    st.metric("Total Revenue", f"${total_rev:,.0f}")

with col3:
    avg_days = df_filtered['actual_days'].mean()
    st.metric("Avg Delivery Time", f"{avg_days:.1f} days")

with col4:
    # Safe division for late rate
    if len(df_filtered) > 0:
        late_pct = (df_filtered['is_late'].sum() / len(df_filtered)) * 100
    else:
        late_pct = 0.0
    
    # Color code: Red if > 10% late, Green if < 10%
    st.metric("Late Delivery Rate", f"{late_pct:.1f}%", delta_color="inverse")

st.divider()

# ROW 2: MAP & CHARTS
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📍 Seller Geography")
    # Map now uses the Filtered Data
    map_data = df_filtered.dropna(subset=['lat', 'lon'])
    st.map(map_data)

with col_right:
    st.subheader("⚠️ Worst Categories")
    # Group by category and calculate average delay
    category_delay = df_filtered.groupby('product_category_name')['delay_days'].mean().reset_index()
    # Sort by worst delay and take top 10
    top_delay = category_delay.sort_values(by='delay_days', ascending=False).head(10)
    
    fig = px.bar(top_delay, x='delay_days', y='product_category_name', orientation='h',
                 title="Avg Delay by Category (Days)", color='delay_days', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# RAW DATA VIEW (Collapsible)
with st.expander("🔎 View Raw Data"):
    st.dataframe(df_filtered.head(100))