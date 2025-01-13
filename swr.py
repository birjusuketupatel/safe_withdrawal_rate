import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
Given a sequence of returns, calculates the t year safe withdrawal rate for each year,
assuming that the investor wants to have c% of their wealth remaining after t years.
"""
def calc_swr(returns, t, c):
    S = []

    for i in range(len(returns) - t + 1):
        # calculate SWR for each t year slice
        slice = returns[i:i + t]

        cum_prod = 1
        for r in slice:
            cum_prod *= r
        numerator = -c + cum_prod

        sum_prod = 0
        for i in range(t):
            cum_prod = 1
            for j in range(i, t):
                cum_prod *= slice[j]

            sum_prod += cum_prod

        swr = numerator / sum_prod
        S.append(swr)

    return S

def main():
    # generate historical safe withdrawal rates for S&P 500 using real and nominal returns
    df = pd.read_csv('data/s&p_returns.csv')
    df = df[df['cape'].notna()]

    t = 30
    residual = 1
    current_cape = 37

    years = df['year'].tolist()
    cape = df['cape'].tolist()
    nominal = df['nominal_return'].tolist()
    real = df['real_return'].tolist()

    real_swr = calc_swr(real, t, residual)
    nominal_swr = calc_swr(nominal, t, residual)

    years = years[:-t + 1]
    cape = cape[:-t + 1]
    cape_yield = [1 / x for x in cape]

    # plot real safe withdrawal rates with CAPE yield
    plt.figure(figsize=(10, 6))
    plt.plot(years, real_swr, marker='o', linestyle='-', color='b')
    plt.plot(years, cape_yield, marker='^', linestyle='-', color='g')

    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.title('Real Safe Withdrawal Rate Over Time')

    plt.legend(['30 Year Safe Withdrawal Rate', 'Shiller Earnings Yield'])

    plt.xticks(np.arange(min(years), max(years) + 1, 10))

    plt.show()

    # plot nominal safe withdrawal rates with CAPE yield
    plt.figure(figsize=(10, 6))
    plt.plot(years, nominal_swr, marker='o', linestyle='-', color='r')
    plt.plot(years, cape_yield, marker='^', linestyle='-', color='g')

    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.title('Nominal Safe Withdrawal Rate Over Time')

    plt.legend(['30 Year Safe Withdrawal Rate', 'Shiller Earnings Yield'])

    plt.xticks(np.arange(min(years), max(years)+1, 10))

    plt.show()

    # regress one over the real and nominal safe withdrawal rates on the CAPE yield
    #real_swr = [1 / r for r in real_swr]
    #nominal_swr = [1 / r for r in nominal_swr]
    x = np.array(cape_yield)
    y_r = np.array(real_swr)
    y_n = np.array(nominal_swr)

    x_intercept = sm.add_constant(x)

    # fit models to data
    real_model = sm.OLS(y_r, x_intercept)
    nominal_model = sm.OLS(y_n, x_intercept)

    real_result = real_model.fit()
    nominal_result = nominal_model.fit()

    # get test data to validate model
    df_test = pd.read_csv('data/ftse_100_returns.csv')

    cape_test = df_test['cape'].tolist()
    nominal_test = df_test['nominal_return'].tolist()
    real_test = df_test['real_return'].tolist()

    real_swr_test = calc_swr(real_test, t, residual)
    nominal_swr_test = calc_swr(nominal_test, t, residual)

    cape_test = cape_test[:-t + 1]
    cape_yield_test = [1 / x for x in cape_test]

    x_test = np.array(cape_yield_test)
    y_r_test = np.array(real_swr_test)
    y_n_test = np.array(nominal_swr_test)

    # use models to establish prediction line and prediction interval
    current_cape_yield = 1 / current_cape
    x_pred = np.linspace(min(current_cape_yield - 0.03, min(x)), max(current_cape_yield + 0.03, max(x)), 100)
    x_pred_intercept = sm.add_constant(x_pred)

    real_pred = real_result.get_prediction(x_pred_intercept)
    nominal_pred = nominal_result.get_prediction(x_pred_intercept)

    # get prediction interval at a 68% confidence level
    real_summary = real_pred.summary_frame(alpha=0.32)
    nominal_summary = nominal_pred.summary_frame(alpha=0.32)

    real_low = real_summary['obs_ci_lower']
    real_upper = real_summary['obs_ci_upper']
    nominal_low = nominal_summary['obs_ci_lower']
    nominal_upper = nominal_summary['obs_ci_upper']

    # plot data, regression line, and prediction interval
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_r, color='b', label='Data Points')
    plt.scatter(x_test, y_r_test, color='orange', label='Test Points')
    plt.plot(x_pred, real_result.predict(x_pred_intercept), color='r', label='Regression Line')
    plt.fill_between(x_pred, real_low, real_upper, color='gray', alpha=0.3, label='Prediction Interval')
    plt.axvline(x=current_cape_yield, color='red', linestyle='--', label='Current Shiller Earnings Yield')

    plt.xlabel('Shiller Earnings Yield')
    plt.ylabel('r')
    plt.title('Regression Model, Real')
    plt.legend()

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_n, color='b', label='Data Points')
    plt.scatter(x_test, y_n_test, color='orange', label='Test Points')
    plt.plot(x_pred, nominal_result.predict(x_pred_intercept), color='r', label='Regression Line')
    plt.fill_between(x_pred, nominal_low, nominal_upper, color='gray', alpha=0.3, label='Prediction Interval')
    plt.axvline(x=current_cape_yield, color='red', linestyle='--', label='Current Shiller Earnings Yield')

    plt.xlabel('Shiller Earnings Yield')
    plt.ylabel('r')
    plt.title('Regression Model, Nominal')
    plt.legend()

    plt.show()

    # print summary statistics for model
    print('\n\nReal Model:')
    print(real_result.summary())

    print('\n\nNominal Model:')
    print(nominal_result.summary())

    # calculate range of possible real and nominal safe withdrawal rates given today's Shiller PE
    real_pred = real_result.get_prediction([1, current_cape_yield])
    nominal_pred = nominal_result.get_prediction([1, current_cape_yield])

    real_swr = 100 * real_pred.predicted_mean[0]
    nominal_swr = 100 * nominal_pred.predicted_mean[0]

    real_summary = real_pred.summary_frame(alpha=0.32)
    nominal_summary = nominal_pred.summary_frame(alpha=0.32)

    real_low  = 100 * real_summary['obs_ci_lower'][0]
    real_high = 100 * real_summary['obs_ci_upper'][0]
    nominal_low = 100 * nominal_summary['obs_ci_lower'][0]
    nominal_high = 100 * nominal_summary['obs_ci_upper'][0]

    print('\nModel Projections:')
    print('\nNominal Model:')
    print('Predicted Safe Withdrawal Rate: {:0.2f}%'.format(nominal_swr))
    print('68% Prediction Interval: [{:0.2f}%, {:0.2f}%]'.format(nominal_low, nominal_high))

    print('\nReal Model:')
    print('Predicted Safe Withdrawal Rate: {:0.2f}%'.format(real_swr))
    print('68% Prediction Interval: [{:0.2f}%, {:0.2f}%]'.format(real_low, real_high))

if __name__ == "__main__":
    main()
