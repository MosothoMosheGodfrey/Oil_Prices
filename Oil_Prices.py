import subprocess
import sys

# List of required libraries
REQUIRED_LIBRARIES = [
    "yfinance",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "statsmodels",
    "prophet",
    "tensorflow",
]

def install_libraries():
    """Install required libraries if they are not already installed."""
    for library in REQUIRED_LIBRARIES:
        try:
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            print(f"{library} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print(f"{library} has been successfully installed.")

# Install required libraries before proceeding
install_libraries()

# Rest of your script starts here
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0 = all logs, 3 = no logs)

# Redirect stdout and stderr to suppress TensorFlow output
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Import TensorFlow after redirecting stdout/stderr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Restore stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

# Suppress cmdstanpy logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Step 1: Fetch historical oil price data
def get_oil_price(symbol, name, start_date):
    """Retrieves oil price data from Yahoo Finance."""
    try:
        data = yf.download(symbol, start=start_date, progress=False)  # Disable progress bar
        if data.empty:
            return None

        # Add 'Product' column after 'Date'
        data.reset_index(inplace=True)
        data.insert(1, 'Product', name)
        return data

    except Exception:
        return None

# Step 2: Process the data
def process_data(data):
    """Processes the merged oil price data."""
    data['Date'] = pd.to_datetime(data['Date'])
    date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
    unique_products = data['Product'].unique()
    complete_data = pd.DataFrame(
        [(date, product) for date in date_range for product in unique_products],
        columns=['Date', 'Product']
    )
    processed_data = pd.merge(complete_data, data, on=['Date', 'Product'], how='left')
    processed_data['close_interpolate'] = processed_data.groupby('Product')['Close'].transform(
        lambda x: x.interpolate(method='linear')
    )
    processed_data['close_interpolate'] = processed_data['close_interpolate'].combine_first(processed_data['Close'])
    processed_data['close_monthlyavg'] = processed_data.groupby(
        [processed_data['Date'].dt.to_period('M'), 'Product']
    )['close_interpolate'].transform('mean')
    processed_data.rename(
        columns={
            'Date': 'InformationDate',
            'Close': 'OilPrice_USD',
            'close_interpolate': 'OilPrice_USD_Interpolate',
            'close_monthlyavg': 'OilPrice_USD_MonthlyAver'
        },
        inplace=True
    )
    processed_data = processed_data[
        ['InformationDate', 'Product', 'OilPrice_USD', 'OilPrice_USD_Interpolate', 'OilPrice_USD_MonthlyAver']
    ]
    processed_data.sort_values(by='InformationDate', inplace=True)
    return processed_data

# Step 3: Forecast oil prices using multiple models
def forecast_oil_prices(processed_data):
    """Forecasts oil prices for the next 8 days."""
    last_date = processed_data['InformationDate'].max()
    if last_date.weekday() > 5:
        return None

    start_date = last_date + timedelta(days=1)
    forecast_data = pd.DataFrame()

    for product in processed_data['Product'].unique():
        product_data = processed_data[processed_data['Product'] == product].copy()
        prophet_data = product_data[['InformationDate', 'OilPrice_USD_Interpolate']].rename(
            columns={'InformationDate': 'ds', 'OilPrice_USD_Interpolate': 'y'}
        ).dropna()

        # Model 1: Prophet
        try:
            model_prophet = Prophet()
            model_prophet.fit(prophet_data)
            future_dates = model_prophet.make_future_dataframe(periods=8, freq='D')
            forecast_prophet = model_prophet.predict(future_dates)
            forecast_prophet = forecast_prophet[['ds', 'yhat']].rename(columns={'yhat': 'Forecast01'})
        except Exception:
            forecast_prophet = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast01': np.nan})

        # Model 2: ARIMA
        try:
            arima_data = prophet_data.set_index('ds')['y']
            model_arima = ARIMA(arima_data, order=(5, 1, 0))
            model_arima_fit = model_arima.fit()
            forecast_arima = model_arima_fit.forecast(steps=8)
            forecast_arima = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast02': forecast_arima
            })
        except Exception:
            forecast_arima = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast02': np.nan})

        # Model 3: SARIMA
        try:
            model_sarima = SARIMAX(arima_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 7))
            model_sarima_fit = model_sarima.fit()
            forecast_sarima = model_sarima_fit.forecast(steps=8)
            forecast_sarima = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast03': forecast_sarima
            })
        except Exception:
            forecast_sarima = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast03': np.nan})

        # Model 4: Exponential Smoothing (Holt-Winters)
        try:
            model_ets = ExponentialSmoothing(arima_data, trend='add', seasonal='add', seasonal_periods=7)
            model_ets_fit = model_ets.fit()
            forecast_ets = model_ets_fit.forecast(steps=8)
            forecast_ets = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast04': forecast_ets
            })
        except Exception:
            forecast_ets = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast04': np.nan})

        # Model 5: LSTM
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(arima_data.values.reshape(-1, 1))
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model_lstm = Sequential()
            model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model_lstm.add(LSTM(50, return_sequences=False))
            model_lstm.add(Dense(25))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X, y, batch_size=32, epochs=1, verbose=0)

            inputs = arima_data[-60:].values
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)
            X_test = []
            X_test.append(inputs[0:60, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_prices = model_lstm.predict(X_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            forecast_lstm = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast05': np.round(np.repeat(predicted_prices[0][0], 8), 3)
            })
        except Exception:
            forecast_lstm = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast05': np.nan})

        # Model 6: GRU
        try:
            model_gru = Sequential()
            model_gru.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model_gru.add(GRU(50, return_sequences=False))
            model_gru.add(Dense(25))
            model_gru.add(Dense(1))
            model_gru.compile(optimizer='adam', loss='mean_squared_error')
            model_gru.fit(X, y, batch_size=32, epochs=1, verbose=0)

            predicted_prices_gru = model_gru.predict(X_test)
            predicted_prices_gru = scaler.inverse_transform(predicted_prices_gru)

            forecast_gru = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast06': np.round(np.repeat(predicted_prices_gru[0][0], 8), 3)
            })
        except Exception:
            forecast_gru = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast06': np.nan})

        # Model 7: XGBoost
        try:
            model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
            model_xgb.fit(X.reshape(X.shape[0], -1), y)
            predicted_prices_xgb = model_xgb.predict(X_test.reshape(X_test.shape[0], -1))
            predicted_prices_xgb = scaler.inverse_transform(predicted_prices_xgb.reshape(-1, 1))

            forecast_xgb = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast07': np.round(np.repeat(predicted_prices_xgb[0][0], 8), 3)
            })
        except Exception:
            forecast_xgb = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast07': np.nan})

        # Model 8: LightGBM
        try:
            model_lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
            model_lgbm.fit(X.reshape(X.shape[0], -1), y)
            predicted_prices_lgbm = model_lgbm.predict(X_test.reshape(X_test.shape[0], -1))
            predicted_prices_lgbm = scaler.inverse_transform(predicted_prices_lgbm.reshape(-1, 1))

            forecast_lgbm = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast08': np.round(np.repeat(predicted_prices_lgbm[0][0], 8), 3)
            })
        except Exception:
            forecast_lgbm = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast08': np.nan})

        # Model 9: Random Forest Regressor
        try:
            model_rf = RandomForestRegressor(n_estimators=100, max_depth=3)
            model_rf.fit(X.reshape(X.shape[0], -1), y)
            predicted_prices_rf = model_rf.predict(X_test.reshape(X_test.shape[0], -1))
            predicted_prices_rf = scaler.inverse_transform(predicted_prices_rf.reshape(-1, 1))

            forecast_rf = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast09': np.round(np.repeat(predicted_prices_rf[0][0], 8), 3)
            })
        except Exception:
            forecast_rf = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast09': np.nan})

        # Model 10: Support Vector Regression (SVR)
        try:
            model_svr = SVR(kernel='rbf')
            model_svr.fit(X.reshape(X.shape[0], -1), y)
            predicted_prices_svr = model_svr.predict(X_test.reshape(X_test.shape[0], -1))
            predicted_prices_svr = scaler.inverse_transform(predicted_prices_svr.reshape(-1, 1))

            forecast_svr = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast10': np.round(np.repeat(predicted_prices_svr[0][0], 8), 3)
            })
        except Exception:
            forecast_svr = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast10': np.nan})

        # Model 11: VAR (Vector AutoRegression)
        try:
            var_data = processed_data.pivot(index='InformationDate', columns='Product', values='OilPrice_USD_Interpolate').dropna()
            model_var = VAR(var_data)
            model_var_fit = model_var.fit()
            forecast_var = model_var_fit.forecast(var_data.values, steps=8)
            forecast_var = pd.DataFrame(forecast_var, columns=var_data.columns)
            forecast_var['ds'] = pd.date_range(start=start_date, periods=8, freq='D')
            forecast_var = forecast_var.melt(id_vars='ds', var_name='Product', value_name='Forecast11')
        except Exception:
            forecast_var = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast11': np.nan})

        # Model 12: Hybrid Model (LSTM + ARIMA)
        try:
            forecast_hybrid = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=8, freq='D'),
                'Forecast12': np.round((forecast_lstm['Forecast05'].values[:8] + forecast_arima['Forecast02'].values[:8]) / 2, 3)
            })
        except Exception:
            forecast_hybrid = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=8, freq='D'), 'Forecast12': np.nan})

        # Merge forecasts
        forecast = pd.merge(forecast_prophet, forecast_arima, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_sarima, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_ets, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_lstm, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_gru, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_xgb, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_lgbm, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_rf, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_svr, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_var, on='ds', how='outer')
        forecast = pd.merge(forecast, forecast_hybrid, on='ds', how='outer')

        # Add product information
        forecast['Product'] = product
        forecast.rename(columns={'ds': 'InformationDate'}, inplace=True)
        forecast_data = pd.concat([forecast_data, forecast], ignore_index=True)

    forecast_data = forecast_data[forecast_data['InformationDate'] > last_date]
    forecast_data = forecast_data[
        ['InformationDate', 'Product', 'Forecast01', 'Forecast02', 'Forecast03', 'Forecast04', 'Forecast05',
         'Forecast06', 'Forecast07', 'Forecast08', 'Forecast09', 'Forecast10', 'Forecast11', 'Forecast12']
    ]
    return forecast_data

# Main function
def main():
    oil_types = {
        "CL=F": "WTI Crude Oil (NYMEX)",
        "BZ=F": "Brent Crude Oil (ICE)",
        "HO=F": "Heating Oil (NYMEX)",
    }
    start_date = "1990-01-01"

    all_data = []
    for symbol, name in oil_types.items():
        oil_data = get_oil_price(symbol, name, start_date)
        if oil_data is not None:
            all_data.append(oil_data)

    merged_data = pd.concat(all_data, ignore_index=True)
    processed_data = process_data(merged_data)
    processed_data.to_csv("Oil_Prices_Processed.csv", index=False)

    forecast_data = forecast_oil_prices(processed_data)
    if forecast_data is not None:
        current_day = datetime.now().strftime("%A")
        forecast_filename = f"{current_day}_Forecast.csv"
        forecast_data.to_csv(forecast_filename, index=False)

if __name__ == "__main__":
    main()
