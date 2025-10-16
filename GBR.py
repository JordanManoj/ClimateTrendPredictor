import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load dataset
data = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
data = data[['dt', 'AverageTemperature', 'Country']]
data['dt'] = pd.to_datetime(data['dt'])
data['AverageTemperature'] = data['AverageTemperature'].interpolate(method='linear')
data['Year'] = data['dt'].dt.year
data['Month'] = data['dt'].dt.month

# Aggregate yearly global average
yearly_temp = data.groupby('Year')['AverageTemperature'].mean().reset_index()

# Remove outliers
z_scores = np.abs(stats.zscore(yearly_temp['AverageTemperature']))
yearly_temp = yearly_temp[z_scores < 3]

# Add lag feature
yearly_temp['Temp_Lag1'] = yearly_temp['AverageTemperature'].shift(1)
yearly_temp = yearly_temp.dropna()

# Features and target
X = yearly_temp[['Year', 'Temp_Lag1']]
y = yearly_temp['AverageTemperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features for models that need it (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R² Score: {r2:.3f}\n")

evaluate(y_test, y_pred_gbr, "Gradient Boosting Regressor")

# Predict future (next 30 years)
future_years = pd.DataFrame({'Year': np.arange(yearly_temp['Year'].max()+1, yearly_temp['Year'].max()+31)})
last_temp = yearly_temp['AverageTemperature'].iloc[-1]
future_years['Temp_Lag1'] = last_temp

future_pred = gbr_model.predict(future_years)

# Combine predictions
future_df = pd.DataFrame({'Year': future_years['Year'], 'PredictedTemperature': future_pred})

# Plot
plt.figure(figsize=(10,5))
sns.lineplot(data=yearly_temp, x='Year', y='AverageTemperature', label='Historical Data')
sns.lineplot(data=future_df, x='Year', y='PredictedTemperature', label='Predicted Data', color='red')
plt.title('Future Global Temperature Forecast with Gradient Boosting')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
