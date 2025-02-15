from datetime import timedelta, datetime
from pathlib import Path
from typing import Callable
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from mplcursors import cursor
import tickers


class StartDateDateMissing(Exception):
    """
    Historical ticker data only starts after the requested `start_date`.
    """

    pass


def download_cached(
    cache_file: str, downloader: Callable[[], pd.DataFrame], *args, **kwargs
) -> pd.DataFrame:
    """
    Calls `downloader(*args, **kwargs)` and saves the result to `cache_file`
    or reads from `cache_file` if it already exists.
    """
    cache = Path(f"cache/{cache_file}.pqt")
    if cache.is_file():
        data = pd.read_parquet(cache)
    else:
        print(f"Cache for {cache_file} not found. Downloading...")
        data = downloader(*args, **kwargs)
        Path("cache").mkdir(exist_ok=True)
        data.to_parquet(cache)
    return data


def get_inflation_adjusted_monthly(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.Series:
    """
    Returns an inflation-adjusted monthly series for `ticker`.
    """
    # Download the ticker data
    fund_data = download_cached(
        f"yf-{ticker} {start_date} {end_date}",
        yf.download,
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1mo",
    )

    # Remove the ticker index level
    fund_data.columns = fund_data.columns.get_level_values(0)

    # Ensure data starts reasonably close to start_date
    fund_start_date = fund_data.index[0]
    if fund_start_date > start_date + timedelta(days=30):
        raise StartDateDateMissing(
            f"{ticker} starts on {fund_start_date} which is over a month later than {start_date}"
        )

    # Download inflation data
    cpi_data = download_cached(
        f"fred-CPIAUCSL {start_date} {end_date}",
        pdr.get_data_fred,
        "CPIAUCSL",
        start=start_date,
        end=end_date,
    )

    # Align inflation data to the mutual fund data
    cpi_data = cpi_data.reindex(fund_data.index, method="nearest", limit=1)

    # Inflation-adjust the ticker data
    base_cpi = cpi_data["CPIAUCSL"].iloc[0]
    inflation_adjusted: pd.Series = fund_data["Close"] / cpi_data["CPIAUCSL"] * base_cpi

    # Double check to ensure there are no NaN
    if inflation_adjusted.isnull().values.any():
        raise Exception(f"Some values in {ticker} data came out to NaN.")

    return inflation_adjusted


def get_rolling_returns(series: pd.Series, rolling_months: int) -> pd.Series:
    """
    Returns the rolling returns of `series` with a window of `rolling_months`.

    Assumes the `series` elements are monthly.
    """
    # Get the rolling returns
    rolling_returns = series.pct_change(periods=rolling_months)

    # Annualize them
    rolling_returns = (rolling_returns + 1) ** (12.0 / rolling_months) - 1
    return rolling_returns


def get_percentile_rolling_returns(
    series: pd.Series, min_month_diff: int, max_month_diff: int, percentile: int = 0.0
) -> pd.Series:
    """
    Returns the `percentile` of all rolling returns in `series` for window sizes from `min_month_diff` to `max_month_diff`.

    Pass `percentile = 0` for the minimum rolling return, `0.5` for median, and `1.0` for the maximum return.

    Assumes the `series` elements are monthly.
    """
    min_returns = []

    for rolling_months in range(min_month_diff, max_month_diff + 1):
        min_return = get_rolling_returns(
            series, rolling_months=rolling_months
        ).quantile(percentile)
        min_returns.append(min_return)

    series = pd.Series(
        data=min_returns, index=range(min_month_diff, max_month_diff + 1)
    )

    return series


def print_summary(
    percentile_rolling_returns: pd.DataFrame, months_period: int, percentile: int = 0.0
) -> None:
    best = percentile_rolling_returns.loc[months_period].nlargest(30)
    worst = percentile_rolling_returns.loc[months_period].nsmallest(3)

    print(
        f"\nOver {months_period} month periods, the {percentile_rolling_returns.name} are:"
    )
    for i, (col, value) in enumerate(best.items()):
        print(f"  {i+1:3}. {value:.2%} - {col}")

    print(f"...")
    total = len(percentile_rolling_returns.columns)
    for i, (col, value) in reversed(list(enumerate(worst.items()))):
        print(f"  {total-i}. {value:.2%} - {col}")


if __name__ == "__main__":

    series_list = []

    percentile = 0.0

    for ticker in tickers.fidelity_funds:
        try:
            fund_data = get_inflation_adjusted_monthly(
                ticker, start_date=datetime(1999, 8, 1), end_date=datetime(2024, 10, 1)
            )
        except StartDateDateMissing as e:
            print(f"Skipping {ticker} because {e}")
            continue

        min_returns = get_percentile_rolling_returns(fund_data, 12, 180, percentile)

        info = yf.Ticker(ticker).info
        long_name = info["longName"]
        try:
            expense_ratio = info["annualReportExpenseRatio"]
        except:
            print(f"{ticker} didn't list expense ratio! Skipping.")
            continue

        min_returns.name = (
            f"{ticker} (expense ratio: {expense_ratio:.2%}) - {long_name}"
        )
        series_list.append(min_returns)

    df = pd.concat(series_list, axis=1)
    df.name = (
        f"{percentile:.2%}th percentile annualized inflation-adjusted rolling returns"
    )

    print_summary(df, 120, percentile)
    print_summary(df, 180, percentile)

    (df * 100).plot()
    plt.xlabel("Rolling return window (months)")
    plt.ylabel(f"{percentile:.2%}th percentile returns")
    plt.grid()
    cursor(hover=True)
    plt.show()
