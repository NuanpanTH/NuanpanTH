import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/pangiiez/workspace/prices.csv',index_col=0)
data = data.iloc[:, [0,1]]
data.index = pd.to_datetime(data.index)
data = data.pivot(columns='symbol', values='close')
data = data[['AAPL','FB','NFLX','V','XOM']]
print(data.head())

fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

prices = pd.read_csv('/Users/pangiiez/workspace/prices.csv',index_col=0)
prices = prices.iloc[:, [0,1]]
prices.index = pd.to_datetime(prices.index)
prices = prices.pivot(columns='symbol', values='close')
prices= prices[['EBAY','YHOO','NVDA','AAPL']]
prices.plot.scatter('EBAY','YHOO',c=prices.index,
                    cmap=plt.cm.viridis,colorbar=False)
plt.show()

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

all_prices = pd.read_csv('/Users/pangiiez/workspace/prices.csv',index_col=0)
all_prices = all_prices.iloc[:, [0,1]]
all_prices.index = pd.to_datetime(all_prices.index)
all_prices = all_prices.pivot(columns='symbol', values='close')
print(all_prices.head())

X = all_prices[['EBAY', 'NVDA', 'YHOO']]
y = all_prices[['AAPL']]

scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()

df = pd.read_csv("/Users/pangiiez/workspace/TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Volume'] = df['Volume'].astype(float)
df['Gain'] = df['Close'] - df['Open']

# Window Rolling Mean (Moving Average)
df['Rolling Close Average'] = df['Close'].rolling(2).mean()

df['Rolling Close Average'].plot()
plt.show()

df['Open Standard Deviation'] = df['Open'].std()
df['Rolling Open Standard Deviation'] = df['Open'].rolling(2).std()
print(df[['Date', 'Open', 'Open Standard Deviation', 'Rolling Open Standard Deviation']].head())
df.plot(x='Date', y='Rolling Open Standard Deviation',legend=False)
plt.tight_layout()
plt.show()

df['Rolling Volume Sum'] = df['Volume'].rolling(3).sum()
print(df[['Date','Volume', 'Rolling Volume Sum']].head())

def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1] # all but the last value
    last_value = series[-1] # last value 

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)

import numpy as np

# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r',lw=2)
ax.legend(loc='best')
plt.show()