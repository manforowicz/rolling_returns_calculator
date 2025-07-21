from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from mplcursors import cursor
from package.fetcher import StartDateMissing, get_inflation_adjusted_monthly
import package.tickers as tickers


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
        except StartDateMissing as e:
            print(f"Skipping {ticker} because {e}")
            continue

        min_returns = get_percentile_rolling_returns(fund_data, 12, 180, percentile)

        min_returns.name = fund_data.name
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
