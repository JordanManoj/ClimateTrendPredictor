import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load dataset
data = pd.read_csv("GlobalLandTemperaturesByCountry.csv")

# Keep relevant columns
data = data[['dt', 'AverageTemperature', 'Country']]

# Convert date to datetime
data['dt'] = pd.to_datetime(data['dt'])

# Interpolate missing values instead of dropping
data['AverageTemperature'] = data['AverageTemperature'].interpolate(method='linear')

# Extract year and month
data['Year'] = data['dt'].dt.year
data['Month'] = data['dt'].dt.month

# Aggregate to yearly global average
yearly_temp = data.groupby('Year')['AverageTemperature'].mean().reset_index()

# Remove outliers using z-score
z_scores = np.abs(stats.zscore(yearly_temp['AverageTemperature']))
yearly_temp = yearly_temp[z_scores < 3]  # Keep rows within 3 standard deviations

# Feature engineering: add a lag feature (previous year temp)
yearly_temp['Temp_Lag1'] = yearly_temp['AverageTemperature'].shift(1)
yearly_temp = yearly_temp.dropna()  # Drop first row due to NaN lag

# Split features and target
X = yearly_temp[['Year', 'Temp_Lag1']]
y = yearly_temp['AverageTemperature']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R² Score: {r2:.3f}\n")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# Future prediction (next 30 years)
future_years = pd.DataFrame({'Year': np.arange(yearly_temp['Year'].max()+1, yearly_temp['Year'].max()+31)})
# Use last known lag value for predictions
last_temp = yearly_temp['AverageTemperature'].iloc[-1]
future_years['Temp_Lag1'] = last_temp

# Scale future features for LR
future_scaled = scaler.transform(future_years)

# Predict using Random Forest (better for trends)
future_pred = rf_model.predict(future_years)

# Combine predictions
future_df = pd.DataFrame({'Year': future_years['Year'], 'PredictedTemperature': future_pred})

# Plot
plt.figure(figsize=(10,5))
sns.lineplot(data=yearly_temp, x='Year', y='AverageTemperature', label='Historical Data')
sns.lineplot(data=future_df, x='Year', y='PredictedTemperature', label='Predicted Data', color='red')
plt.title('Future Global Temperature Forecast')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()