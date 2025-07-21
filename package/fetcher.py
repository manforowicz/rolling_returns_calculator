from datetime import timedelta, datetime
from pathlib import Path
from typing import Callable, Any
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd


class StartDateMissing(Exception):
    """
    Historical ticker data only starts after the requested `start_date`.
    """

    pass


def download_cached(
    cache_file: str, downloader: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    """
    Calls `downloader()` and saves the result to `cache_file`
    or reads from `cache_file` if it already exists.
    """
    cache = Path(f"cache/{cache_file}.pqt")
    if cache.is_file():
        data = pd.read_parquet(cache)
    else:
        print(f"Cache for {cache_file} not found. Downloading...")
        data = downloader()
        Path("cache").mkdir(exist_ok=True)
        data.to_parquet(cache)
    return data


def get_inflation_adjusted_monthly(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.Series:
    """
    Returns an inflation-adjusted monthly series for `ticker`.
    """

    def yahoo_download():
        # Download price data
        data = yf.download(
            tickers=ticker, start=start_date, end=end_date, interval="1mo"
        )
        if data is None:
            raise RuntimeError("yf returned None")

        # Remove the ticker index level
        data.columns = data.columns.get_level_values(0)

        # Download metadata
        info = yf.Ticker(ticker).info
        long_name = info.get("longName")
        if long_name is None:
            long_name = ticker

        expense_ratio = info.get("annualReportExpenseRatio")
        if expense_ratio is None:
            expense_ratio = float("nan")

        data.attrs["Description"] = (
            f"{ticker} (expense ratio: {expense_ratio:.2%}) - {long_name}"
        )

        return data

    # Download the ticker data
    fund_data = download_cached(f"yf-{ticker} {start_date} {end_date}", yahoo_download)

    # Download inflation data
    cpi_data = download_cached(
        f"fred-CPIAUCSL {start_date} {end_date}",
        lambda: pdr.get_data_fred("CPIAUCSL", start=start_date, end=end_date),
    )

    # Ensure data starts reasonably close to start_date
    fund_start_date = fund_data.index[0]
    if fund_start_date > start_date + timedelta(days=30):
        raise StartDateMissing(
            f"{ticker} starts on {fund_start_date} which is over a month later than {start_date}"
        )

    # Align inflation data to the mutual fund data
    cpi_data = cpi_data.reindex(fund_data.index, method="nearest", limit=1)

    # Inflation-adjust the ticker data
    base_cpi = cpi_data["CPIAUCSL"].iloc[0]
    inflation_adjusted: pd.Series = fund_data["Close"] / cpi_data["CPIAUCSL"] * base_cpi

    inflation_adjusted.name = fund_data.attrs["Description"]

    # Double check to ensure there are no NaN
    if inflation_adjusted.isnull().values.any():
        raise Exception(f"Some values in {ticker} data came out to NaN.")

    return inflation_adjusted
