# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
## DEVELOPED BY:A.Sasidharan
## REGISTER NO: 212221240049
## DATE:


### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/globaltemper.csv')

# Convert 'dt' to datetime and set as index
# The error was due to incorrect date format. Specify the correct format using dayfirst=True or format='%d-%m-%Y'
data['dt'] = pd.to_datetime(data['dt'], dayfirst=True)  # or format='%d-%m-%Y' if dayfirst doesn't work
data.set_index('dt', inplace=True)

# Plot the time series data for 'AverageTemperature'
plt.plot(data.index, data['AverageTemperature'])
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('Average Temperature Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of 'AverageTemperature'
check_stationarity(data['AverageTemperature'])

# Plot ACF and PACF for 'AverageTemperature'
plot_acf(data['AverageTemperature'])
plt.show()
plot_pacf(data['AverageTemperature'])
plt.show()

# Train-test split (80% train, 20% test) for 'AverageTemperature'
train_size = int(len(data) * 0.8)
train, test = data['AverageTemperature'][:train_size], data['AverageTemperature'][train_size:]

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values for 'AverageTemperature'
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('SARIMA Model Predictions for Average Temperature')
plt.legend()
plt.show()

```
### OUTPUT:
![Screenshot 2024-11-11 112317](https://github.com/user-attachments/assets/028795f8-5f83-4520-a196-dee70d066813)
![Screenshot 2024-11-11 112401](https://github.com/user-attachments/assets/2b07e47e-ff9e-43d3-ac69-df4d9be6798c)
![Screenshot 2024-11-11 112427](https://github.com/user-attachments/assets/53032f6a-72b3-4c87-8553-007f03256021)
![Screenshot 2024-11-11 112453](https://github.com/user-attachments/assets/365a5a87-517f-4958-827f-648fbae827f4)


### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
