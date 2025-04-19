
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")
daily_sales = df.groupby('Order Date')['Sales'].sum().asfreq('D').fillna(0)

train = daily_sales[:-90]
test = daily_sales[-90:]

model = ARIMA(train, order=(5, 1, 0))
result = model.fit()

forecast = result.forecast(steps=90)

plt.figure(figsize=(14, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='orange')
plt.plot(test.index, forecast, label='Forecast', color='green')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
