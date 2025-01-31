

-- Create the table
CREATE TABLE OilPriceForecastModels (
    ModelCode VARCHAR(20) PRIMARY KEY,
    ModelName VARCHAR(255),
    PythonLibrary VARCHAR(255),
    PyPILink TEXT
);

-- Insert data into the table
INSERT INTO OilPriceForecastModels (ModelCode, ModelName, PythonLibrary, PyPILink) VALUES
('Forecast01', 'Prophet (by Facebook/Meta)', 'prophet', 'https://pypi.org/project/prophet/'),
('Forecast02', 'ARIMA (AutoRegressive Integrated Moving Average)', 'statsmodels', 'https://pypi.org/project/statsmodels/'),
('Forecast03', 'SARIMA (Seasonal ARIMA)', 'statsmodels', 'https://pypi.org/project/statsmodels/'),
('Forecast04', 'Exponential Smoothing (Holt-Winters)', 'statsmodels', 'https://pypi.org/project/statsmodels/'),
('Forecast05', 'LSTM (Long Short-Term Memory)', 'TensorFlow, Keras, PyTorch', 'https://pypi.org/project/tensorflow/, https://pypi.org/project/keras/, https://pypi.org/project/torch/'),
('Forecast06', 'GRU (Gated Recurrent Unit)', 'TensorFlow, Keras, PyTorch', 'https://pypi.org/project/tensorflow/, https://pypi.org/project/keras/, https://pypi.org/project/torch/'),
('Forecast07', 'XGBoost (Extreme Gradient Boosting)', 'xgboost', 'https://pypi.org/project/xgboost/'),
('Forecast08', 'LightGBM (Gradient Boosting)', 'lightgbm', 'https://pypi.org/project/lightgbm/'),
('Forecast09', 'Random Forest Regressor', 'scikit-learn', 'https://pypi.org/project/scikit-learn/'),
('Forecast010', 'Support Vector Regression (SVR)', 'scikit-learn', 'https://pypi.org/project/scikit-learn/'),
('Forecast011', 'VAR (Vector AutoRegression)', 'statsmodels', 'https://pypi.org/project/statsmodels/'),
('Forecast012', 'Hybrid Model (LSTM + ARIMA)', 'TensorFlow, Keras, statsmodels', 'https://pypi.org/project/tensorflow/, https://pypi.org/project/keras/, https://pypi.org/project/statsmodels/');
