import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'weather.csv'
data = pd.read_csv(file_path, delimiter=',',header=11,skipinitialspace=True)
data.head(24)

temperature = np.array(data['Temperatute'])

num_periods = 24
f_horizon = 1
x_train = temperature[:(len(temperature)-(num_periods*2))]
x_batches = x_train.reshape(-1, num_periods, 1)

y_train = temperature[1:(len(temperature)-(num_periods*2))+f_horizon]
y_batches = y_train.reshape(-1, num_periods, 1)

def test_data(series, forecast, num):
    testX = temperature[-(num + forecast):][:num].reshape(-1, num_periods, 1)
    testY = temperature[-(num):].reshape(-1, num_periods, 1)
    return testX, testY
X_test, Y_test = test_data(temperature, f_horizon, 24*2)
print(X_test.shape)

plt.title("Compare Weather Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
plt.show()