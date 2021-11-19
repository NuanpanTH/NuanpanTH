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