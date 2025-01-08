# Safe Withdrawal Estimator
#### Birju Patel
Using the Shiller dataset on large cap US stock returns, I construct a simple linear model to predict the safe spending rate for an investor who wishes to harvest a steady income stream from their volatile portfolio.

The model utilizes the fact that long term stock market returns can be forecasted using the Shiller PE ratio. The Shiller PE ratio is a metric commonly used to gauge the level of valuations in the stock market, and its relationship to long term returns was documented by Professors Campbell and Shiller in their 1988 paper "Stock Prices, Earnings, and Expected Dividends".

To use the model, first download this repository. Ensure that Python is installed on your machine. Then run the following commands to install the required dependencies.
```
python -m pip install numpy pandas statsmodels matplotlib
```

Now, you can run the model with the following command.
```
python swr.py
```

The model assumes that the investor wishes to withdraw a constant amount each year for 30 years. Also, at the end of the 30 years, the investor desires to keep 100% of their principal intact. The model requires you to hardcode the current level Shiller PE ratio as well. To change these assumptions, you must modify the following variables.
```
 t = 30
 residual = 1
 current_cape = 37.5
```
t is the number of years, residual is the portion of principal the investor wants to keep after the t years are over, and current_cape is the current Shiller PE ratio for the market.
