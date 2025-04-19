
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
daily_sales = df.groupby('Order Date')['Sales'].sum().asfreq('D').fillna(0)

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(daily_sales.values.reshape(-1, 1))

train_scaled = sales_scaled[:-90]
test_scaled = sales_scaled[-90:]

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(np.concatenate((train_scaled[-30:], test_scaled)), seq_length=30)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)

pred = model.predict(X_test)
pred_rescaled = scaler.inverse_transform(pred)
true_rescaled = scaler.inverse_transform(y_test)

plt.plot(true_rescaled, label='Actual')
plt.plot(pred_rescaled, label='Forecast')
plt.title('LSTM Forecast')
plt.legend()
plt.show()
