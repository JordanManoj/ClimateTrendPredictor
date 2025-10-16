import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy import stats

# Data loading and preprocessing
data = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
data = data[['dt', 'AverageTemperature']]
data['dt'] = pd.to_datetime(data['dt'])

# Interpolate missing values
data['AverageTemperature'] = data['AverageTemperature'].interpolate(method='linear')

# Monthly global average
monthly_temp = data.groupby('dt')['AverageTemperature'].mean().reset_index()

# The outliers are removed using z-score
z_scores = np.abs(stats.zscore(monthly_temp['AverageTemperature']))
monthly_temp = monthly_temp[z_scores < 3]

monthly_temp['Month'] = monthly_temp['dt'].dt.month
monthly_temp['Month_sin'] = np.sin(2 * np.pi * monthly_temp['Month']/12)
monthly_temp['Month_cos'] = np.cos(2 * np.pi * monthly_temp['Month']/12)

# Rolling mean and std (12 months)
monthly_temp['RollingMean_12'] = monthly_temp['AverageTemperature'].rolling(window=12).mean()
monthly_temp['RollingStd_12'] = monthly_temp['AverageTemperature'].rolling(window=12).std()

# Droping rows with NaN values in it
monthly_temp = monthly_temp.dropna().reset_index(drop=True)


# Features and target prepration
features = ['Month_sin', 'Month_cos', 'RollingMean_12', 'RollingStd_12']
target = 'AverageTemperature'

X = monthly_temp[features].values
y = monthly_temp[target].values.reshape(-1,1)

# scaling features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


# Prepareing the sequences for LSTM (the sequence is foe 24 months)
SEQ_LENGTH = 24  
def create_sequences(X, y, seq_length=SEQ_LENGTH):
    X_seq, y_seq = [], []
    for i in range(len(X)-seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Train-test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


# Building the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(SEQ_LENGTH, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training and model evaluvation
model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("Advanced LSTM Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

#R² = 1 → Perfect fit 
#R² = 0 → No predictive power
#R² < 0 → Worse than random guess

# Prediction for the next 30 months
future_preds = []
last_sequence = X_scaled[-SEQ_LENGTH:].copy()

for i in range(30):
    pred_scaled = model.predict(last_sequence.reshape(1, SEQ_LENGTH, X_scaled.shape[1]))
    future_preds.append(pred_scaled[0,0])
    
   
    next_row = last_sequence[-1].copy()
    next_row[2] = pred_scaled[0,0]  
    next_row[3] = 0  
    last_sequence = np.vstack((last_sequence[1:], next_row))

future_preds_actual = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1))
future_dates = pd.date_range(start=monthly_temp['dt'].iloc[-1] + pd.DateOffset(months=1), periods=30, freq='MS')


# Ploting the  results
plt.figure(figsize=(12,6))
plt.plot(monthly_temp['dt'], monthly_temp['AverageTemperature'], label='Historical Data')
plt.plot(future_dates, future_preds_actual, label='Predicted Data', color='red')
plt.title('Future Global Temperature Forecast (Advanced LSTM)')
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()