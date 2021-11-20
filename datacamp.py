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
prices= prices[['EBAY','YHOO']]
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