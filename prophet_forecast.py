import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


def run_prophet_forecast():
    """
    Loads data, trains the Prophet model, and generates a 30-day forecast.
    """
    print("⏳ Loading data and training Prophet model...")
    
    try:
        # Load the time series data
        df = pd.read_csv('data/ts_sales.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/ts_sales.csv not found.")
        return

    # 1. Prepare data for Prophet (MUST rename columns)
    df.columns = ['y']  # Sales Volume
    df.index.name = 'ds' # Date
    df_prophet = df.reset_index()
    
    # 2. Initialize and Train the Model
    # We explicitly add weekly seasonality since we confirmed it in the EDA
    model = Prophet(
        yearly_seasonality=False, # We only have 1.6 years, so yearly is weak
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    
    # Fit the model to the sales history
    model.fit(df_prophet)
    print("✅ Prophet model trained successfully.")

    # 3. Create the Future Dataframe
    # We want to forecast for 30 days beyond the last historical date
    future = model.make_future_dataframe(periods=30)
    
    # 4. Generate the Forecast
    forecast = model.predict(future)

    # 5. Evaluate (Check how well the model predicted the historical data)
    
    # Align historical and predicted data for RMSE calculation
    df_compare = df_prophet.merge(
        forecast[['ds', 'yhat']], 
        on='ds', 
        how='inner'
    )
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(df_compare['y'], df_compare['yhat']))

    print(f"\n--- Evaluation ---")
    print(f"Prediction Error (RMSE): {rmse:.4f}")
    
    # 6. Plot the forecast (saves a visualization)
    fig = model.plot(forecast)
    plt.title(f"30-Day Sales Forecast (RMSE: {rmse:.4f})")
    plt.show() 
    
    # 7. Print the 30-day forecast
    print("\n--- Next 30 Days Forecast (Sales Volume) ---")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_string(index=False))


if __name__ == '__main__':
    run_prophet_forecast()