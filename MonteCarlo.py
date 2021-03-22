import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt 
from pandas_datareader import data as pdr 

# Import our data
def get_stock(stocks, start, end):
    stockdata = pdr.get_data_yahoo(stocks, start, end)
    stockdata = stockdata['Close']
    returns = stockdata.pct_change()
    meanreturns = returns.mean()
    covmatrix = returns.cov()
    return meanreturns, covmatrix

stocklist = ['JPM', 'AAPL', 'MSFT', 'NFLX', 'INTC', 'AMD', 'NVDA']
stocks = [stock for stock in stocklist]
enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=300)

meanreturns, covmatrix = get_stock(stocks, startdate, enddate)

# randomize weights
weights = np.random.random(len(meanreturns))
# Weights add up to 1
weights /= np.sum(weights)


# Monte Carlo Simulations
mc_sims = 400
# Time in days
T = 100

# Arrays to store and retrive information
meanM = np.full(shape=(T, len(weights)), fill_value=meanreturns)
meanM = meanM.T 

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

# Set our portfolio value
initialPortfolio = 10000


for m in range(0, mc_sims):
    # Monte Carlo loops
    normaldist = np.random.normal(size=(T, len(weights)))
    Ltriangle = np.linalg.cholesky(covmatrix)
    # Get our daily returns 
    dailyreturns = meanM + np.inner(normaldist, Ltriangle)

    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyreturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio')
plt.show()