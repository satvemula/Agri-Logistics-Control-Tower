import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def run_decomposition():
    """
    Loads the daily sales data and performs seasonal decomposition.
    """
    print("⏳ Loading time series data and performing decomposition...")
    
    try:
        ts_df = pd.read_csv('data/ts_sales.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/ts_sales.csv not found. Ensure the ETL step completed.")
        return

    # 1. Analyze the time series
    ts_data = ts_df['sales_volume']
    
    # 2. Check basic stats
    print(f"\n--- Basic Sales Statistics ---")
    print(f"Max Daily Sales: {ts_data.max():.0f}")
    print(f"Mean Daily Sales: {ts_data.mean():.2f}")

    # 3. Perform Decomposition
    # We choose a weekly frequency (7 days) as a starting point for seasonality
    decomposition = seasonal_decompose(ts_data, model='additive', period=7)

    # 4. Plot the results (this is for visualization, you will describe the output)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    
    # Plotting each component
    decomposition.observed.plot(ax=axes[0], title='Observed (Sales Volume)')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal (Weekly)')
    decomposition.resid.plot(ax=axes[3], title='Residual (Noise)')
    
    plt.tight_layout()
    plt.show() # Displays the plot
    
    print("\n✅ Decomposition plot displayed. Analyze the 'Trend' and 'Seasonal' components.")


if __name__ == '__main__':
    run_decomposition()
    