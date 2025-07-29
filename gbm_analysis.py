from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

from package import tickers
from package.fetcher import StartDateMissing, get_inflation_adjusted_monthly

# TODO: Instead of assuming log-normal distribution of gains for each month, use the actual
# distribution in the data, multiply it by itself the right number of times, and
# take quantiles of that baby!

def get_gbm_parameters(
    price_data: np.ndarray, points_per_year: int
) -> tuple[float, float]:
    log_returns = np.diff(np.log(price_data))

    # Annualize parameters (12 months a year)
    mu = log_returns.mean() * points_per_year
    sigma = log_returns.std(ddof=1) * np.sqrt(points_per_year)
    return mu, sigma


def get_gbm_quantile(mu: float, sigma: float, quantile: float, years: float) -> float:
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

    return percent_change


def get_kelly_fraction(mu: float, sigma: float) -> float:
    r = 0.0

    return (mu - r) / sigma**2


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

        mu, sigma = get_gbm_parameters(fund_data.to_numpy(), 12)

        low = get_gbm_quantile(mu, sigma, 0.1, 20)
        high = get_gbm_quantile(mu, sigma, 0.5, 20)
        kelly_fraction = get_kelly_fraction(mu, sigma)

        results.append((fund_data.name, low, high, kelly_fraction))

    results.sort(key=lambda x: x[1])
    results.reverse()

    for name, low, high, kelly_fraction in results:
        print(f"{low:.3%} -- {high:.3%} (f*: {kelly_fraction:.3%}): {name}")
