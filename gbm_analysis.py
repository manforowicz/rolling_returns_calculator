from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

from package import tickers
from package.fetcher import StartDateMissing, get_inflation_adjusted_monthly


def get_gbm_quantile(price_data: np.ndarray, points_per_year: int, quantile: float, years: float) -> float:
    log_returns = np.diff(np.log(price_data))

    # Annualize parameters (12 months a year)
    mu = log_returns.mean() * points_per_year
    sigma = log_returns.std(ddof=1) * np.sqrt(points_per_year)

    # Z-score for quantile in a standard normal distribution
    z = stats.norm.ppf(quantile)

    # Compute the log of the terminal price relative to initial price (log-return)
    # According to GBM, log(S_T / S_0) is normally distributed with:
    # mean = (mu - 0.5 * sigma^2) * T
    # std  = sigma * sqrt(T)
    mean_log_return = (mu - 0.5 * sigma**2) * years
    std_log_return = sigma * np.sqrt(years)

    # 5th percentile log-return
    log_return_quantile = mean_log_return + z * std_log_return

    # Convert back from log-return to percent change
    # S_T / S_0 = exp(log_return)
    percent_change = np.exp(log_return_quantile) - 1

    ###### PLOT
    # # Step 3: Plot histogram of the data
    # count, bins, _ = plt.hist(log_returns, bins=20, density=True, alpha=0.6, color='skyblue', label='Data histogram')

    # # Step 4: Overlay the lognormal PDF
    # x = np.linspace(min(log_returns), max(log_returns), 1000)
    # pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    # plt.plot(x, pdf, 'r-', lw=2, label='Lognormal PDF')

    # # Final touches
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Histogram with Lognormal Overlay')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    return percent_change


if __name__ == "__main__":
    results = []

    for ticker in tickers.fidelity_funds:
        try:
            fund_data = get_inflation_adjusted_monthly(
                ticker, start_date=datetime(1999, 8, 1), end_date=datetime(2024, 10, 1)
            )
        except StartDateMissing as e:
            print(f"Skipping {ticker} because {e}")
            continue

        low = get_gbm_quantile(fund_data.to_numpy(), 12, 0.1, 20)
        high = get_gbm_quantile(fund_data.to_numpy(), 12, 0.5, 20)

        results.append((fund_data.name, low, high))

    results.sort(key=lambda x: x[1])
    results.reverse()

    for (name, low, high) in results:
        print(f"{low:.3%} -- {high:.3%}: {name}")