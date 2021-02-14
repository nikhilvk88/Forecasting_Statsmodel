import pandas as pd
import numpy as np


# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima



# Load dataset
df = pd.read_csv('co2_mm_mlo.csv')

#Combine year and date column to get date column
df['date']=pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

# Set "date" to be the index
df.set_index('date',inplace=True)
#montly data
df.index.freq = 'MS'

#Plot source data
title = 'Monthly Mean CO₂ Levels'
ylabel='parts per million'
xlabel=''

ax = df['interpolated'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

#To check if the data indeed has sesonality, use ETS decompose:
result = seasonal_decompose(df['interpolated'], model='add') # or model = 'mul'
result.plot();
#Check how many datapoints(rows) make up a season i.e. to determine m value
result.seasonal.plot(figsize=(12,8))

#Run pmdarima.auto_arima to obtain recommended orders
auto_arima(df['interpolated'],seasonal=True,m=12).summary()


# Set one year for testing
train = df.iloc[:717]
test = df.iloc[717:]

#Use order as obtained from pmdarima
model = SARIMAX(train['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()

# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
#Passing dynamic=False means that forecasts at each point are generated using the full history up to that point (all lagged values).
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(0,1,3)(1,0,1,12) Predictions')

# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['interpolated'][i]}")

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels'
ylabel='parts per million'
xlabel=''

ax = test['interpolated'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

#Evaluate the model:
from sklearn.metrics import mean_squared_error

error = mean_squared_error(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) MSE Error: {error:11.10}')

from statsmodels.tools.eval_measures import rmse

error = rmse(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) RMSE Error: {error:11.10}')

#Retrain the model on the full data, and forecast the future
model = SARIMAX(df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
fcast = results.predict(len(df),len(df)+12-1,typ='levels').rename('SARIMA(0,1,3)(1,0,1,12) Forecast')

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels'
ylabel='parts per million'
xlabel=''

ax = df['interpolated'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);
