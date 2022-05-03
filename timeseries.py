from re import L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('AirPassengers.csv')
df.head()
df.tail()

# convert the month column into a datetime object
df['Month'] = pd.to_datetime(df['Month'], format = '%Y-%m')
# print(df.head())

# convert the month column to an index
df.index = df['Month']
del df['Month']
# print(df.head())
plt.plot(df)
plt.show()
# sns.lineplot(df.index, df['#Passengers'].values)
# plt.ylabel('Number of Passengers')
# plt.show()
# print(df.index)
# print(df['#Passengers'].values)

# let's calculate a seven-month rolling mean

rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()

# print(rolling_mean)
# print(rolling_std)

plt.plot(df, color = 'blue', label = 'Original Passenger Data')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean Passenger Data')
# rolling mean is smoother
plt.plot(rolling_std, color = 'black', label = 'Rolling STD Passenger Data')
# std is vibrating larger
plt.legend(loc="best")

# Let's import augumented Dickey-Fuller test

# Let's pass our data frame into adfuller method
adft = adfuller(df, autolag = 'AIC')
print(adft)

output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", 
                                                        "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
print(output_df)                                                   

# since our p value is >0.05 and null hypothesis is 
# there is no stationarity in the data set

# we see a clear, increasing trend in the # of passengers

autocorrelation_lag1 = df['#Passengers'].autocorr(lag = 1)
autocorrelation_lag3 = df['#Passengers'].autocorr(lag = 3)
autocorrelation_lag6 = df['#Passengers'].autocorr(lag = 6)
autocorrelation_lag9 = df['#Passengers'].autocorr(lag = 9)
print("One Month Lag: ", autocorrelation_lag1)
print("three Month Lag: ", autocorrelation_lag3)
print("six Month Lag: ", autocorrelation_lag6)
print("nine Month Lag: ", autocorrelation_lag9)

psg = np.array(df['#Passengers'])
# print(psg)
psg_temp = psg[1:]
psg_minus_1 = psg[:-1]
print(np.corrcoef(psg_temp,psg_minus_1)[0,1])
# we see that even with a nine month lag, the data is highly autocorrelated
# This further illustration of short and long term trends in the data
auto_list = []
for i in range(1,30):
    auto_list.append(df['#Passengers'].autocorr(lag = i))

# plt.plot(list(range(1,30)),auto_list)
# plt.show()

# print(auto_list)

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['#Passengers'], model = 'additive', period = 7)
decompose.plot()
plt.show()

# from this plot, we can clearly see the increasing trend in number
# of passengers and seasonality patterns in the rise and fall in values each year

df['Date'] = df.index
train = df[df['Date'] < pd.to_datetime("1960-08", format = '%Y-%m')]
train['train'] = train['#Passengers']
print(train)
del train['Date']
del train['#Passengers']
test = df[df['Date'] >=  pd.to_datetime("1960-08", format = '%Y-%m')]
print(test)
del test['Date']
test['test'] = test['#Passengers']
del test['#Passengers']


from pmdarima.arima import auto_arima

model = auto_arima(train, trace = True, error_action = 'ignore', suppress_warnings = True)
model.fit(train)
forecast = model.predict(n_periods = len(test))
forecast = pd.DataFrame(forecast, index = test.index, columns = ['Prediction'])

print(forecast)

plt.plot(train, color = 'black')
plt.plot(test, color = 'red')
plt.plot(forecast, color = 'green')
plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set()
plt.show()


print(test)
print(forecast)