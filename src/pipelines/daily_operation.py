import numpy as np
import pandas as pd

from src.pipelines.best_reserve_market import BATTERY_MARKETS, build_best_market_plans
from src.pipelines.da_arbitrage import (
    PROCESSED_DIR,
    TIMEZONE,
    build_daily_savings,
    read_da_prices,
)
from src.pipelines.reserve_prices import build_reserve_prices

RESERVE = "reserve"
DA = "da"


def build_daily_reserve_yields(plans: pd.DataFrame, battery: str) -> pd.DataFrame:
    """Daily reserve yield (€/MW) of a battery's best market plan, by market.

    Days are Finnish local days. Every hour's yield goes to the market the plan
    picked for that hour, so the columns sum to the day's reserve yield. A day is NaN
    if any of its hours has no price.
    """
    price = plans[f"best_price_{battery}"]
    market = plans[f"best_market_{battery}"]
    hourly = pd.DataFrame(
        {m: price.where(market == m, 0.0) for m in BATTERY_MARKETS[battery]}
    ).where(price.notna())
    days = pd.DatetimeIndex(hourly.index.tz_convert(TIMEZONE).date, name="date")
    grouped = hourly.groupby(days)
    daily = grouped.sum(min_count=1)
    # Local days have 23 or 25 hours when the clocks change
    day_starts = daily.index.tz_localize(TIMEZONE)
    day_ends = (daily.index + pd.Timedelta(days=1)).tz_localize(TIMEZONE)
    hours_in_day = (day_ends - day_starts) / pd.Timedelta("1h")
    complete = grouped.count().eq(hours_in_day, axis=0)
    return daily.where(complete)


def select_operation(reserve: pd.DataFrame, da: pd.Series) -> pd.DataFrame:
    """Choose the better of reserve participation and DA arbitrage for every day.

    `reserve` has the daily yield of each market under the best market plan. Reserve
    is chosen only if its yield exceeds the DA savings, so a tie goes to DA and a day
    without a reserve yield (no market available) is a DA day. The result has the
    chosen mode, the yield of each market and the DA savings. Yields of the operation
    that was not chosen are 0. Days without DA savings are NaN.
    """
    use_reserve = reserve.sum(axis=1, min_count=1).fillna(0) > da
    operation = reserve.fillna(0.0).where(use_reserve, 0.0)
    operation[DA] = da.where(~use_reserve, 0.0)
    operation.insert(0, "mode", np.where(use_reserve, RESERVE, DA))
    return operation.where(da.notna(), axis=0)


def build_daily_operation(prices: pd.DataFrame, da_prices: pd.Series) -> pd.DataFrame:
    """Best daily operation of every battery case.

    Yields are € per MW of battery power. Only days covered by both the reserve and
    the day-ahead data are included.
    """
    plans = build_best_market_plans(prices)
    da = build_daily_savings(da_prices)
    da.index = pd.DatetimeIndex(da.index, name="date")
    operations = []
    for battery in BATTERY_MARKETS:
        reserve = build_daily_reserve_yields(plans, battery)
        days = reserve.index.intersection(da.index)
        operation = select_operation(reserve.loc[days], da.loc[days, f"savings_{battery}"])
        operations.append(operation.add_suffix(f"_{battery}"))
    return pd.concat(operations, axis=1)


def main():
    df = build_daily_operation(build_reserve_prices(), read_da_prices())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "daily_optimal_operation.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")
    for battery in BATTERY_MARKETS:
        print(f"Operation of the {battery} battery:")
        print(df[f"mode_{battery}"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
