import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


def run_prophet_forecast_weekly():
    """
    Loads WEEKLY data, trains the Prophet model, and generates a 4-week forecast.
    """
    print("⏳ Loading WEEKLY data and training Prophet model...")
    
    try:
        # Load the WEEKLY time series data
        df = pd.read_csv('data/ts_sales_weekly.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/ts_sales_weekly.csv not found.")
        return

    # 1. Prepare data for Prophet (MUST rename columns)
    df.columns = ['y']  # Weekly Sales Volume
    df.index.name = 'ds' # Date
    df_prophet = df.reset_index()
    
    # 2. Initialize and Train the Model
    # Since we are using weekly data, the 'weekly_seasonality' is now intrinsic to the data itself, 
    # but we will still look for possible yearly seasonality if the data supports it.
    model = Prophet(
        yearly_seasonality=True, # Look for patterns every 52 weeks
        weekly_seasonality=False, # Now handled by the data aggregation
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    
    model.fit(df_prophet)
    print("✅ Prophet model trained successfully on weekly data.")

    # 3. Create the Future Dataframe (4 WEEKS into the future)
    future = model.make_future_dataframe(periods=4, freq='W')
    
    # 4. Generate the Forecast
    forecast = model.predict(future)

    # 5. Evaluate (Check how well the model predicted the historical data)
    df_compare = df_prophet.merge(
        forecast[['ds', 'yhat']], 
        on='ds', 
        how='inner'
    )
    
    rmse = np.sqrt(mean_squared_error(df_compare['y'], df_compare['yhat']))

    print(f"\n--- Evaluation (Weekly Data) ---")
    print(f"Prediction Error (RMSE): {rmse:.4f}")
    
    # 6. Plot the forecast (saves a visualization)
    fig = model.plot(forecast)
    plt.title(f"4-Week Sales Forecast (RMSE: {rmse:.4f})")
    plt.show() 
    
    # 7. Print the 4-week forecast
    print("\n--- Next 4 Weeks Forecast (Sales Volume) ---")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4).to_string(index=False))


if __name__ == '__main__':
    run_prophet_forecast_weekly()