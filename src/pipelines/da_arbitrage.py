from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TIMEZONE = "Europe/Helsinki"
BATTERY_HOURS = (1, 3)  # Battery durations (energy / power)


def read_da_prices(raw_dir: Path = RAW_DIR) -> pd.Series:
    """Read hourly day-ahead prices (€/MWh) indexed by Finnish local time."""
    df = pd.read_csv(raw_dir / "da_prices_hourly_all.csv", index_col=0)
    prices = df["EUR_MWh_hourly"]
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(TIMEZONE)
    prices.index.name = "start_time"
    if prices.index.duplicated().any():
        raise ValueError("da_prices_hourly_all.csv contains duplicate timestamps")
    return prices.sort_index()


def one_cycle_savings(prices: np.ndarray, battery_hours: int) -> float:
    """Best arbitrage savings (€ per MW of battery power) from at most one full cycle.

    The battery holds `battery_hours` MWh per MW and charges or discharges at the same
    power (1 MW) whatever its duration, i.e. at a C-rate of 1 / `battery_hours`. It
    works in whole hours, so one full cycle is `battery_hours` charging hours and
    `battery_hours` discharging hours. The state of charge stays within
    [0, battery_hours] and the battery is empty at the end of the day. Partial cycles
    are allowed, so the savings are never negative. Losses are not modelled.
    """
    n = battery_hours
    # value[c, d]: best profit after c hours charged and d hours discharged
    value = np.full((n + 1, n + 1), -np.inf)
    value[0, 0] = 0.0
    c_idx, d_idx = np.indices(value.shape)
    negative_soc = d_idx > c_idx
    for price in prices:
        new = value.copy()  # Idle
        new[1:, :] = np.maximum(new[1:, :], value[:-1, :] - price)  # Charge
        new[:, 1:] = np.maximum(new[:, 1:], value[:, :-1] + price)  # Discharge
        new[negative_soc] = -np.inf
        value = new
    return float(np.max(np.diagonal(value)))


def build_daily_savings(prices: pd.Series) -> pd.DataFrame:
    """Day-ahead arbitrage savings (€/MW) for every Finnish local day.

    Days with missing prices or missing hours get NaN.
    """
    days = prices.index.normalize()
    rows = {}
    for day, day_prices in prices.groupby(days):
        # Add a calendar day on the naive date so that DST days have 23 or 25 hours
        next_day = (day.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(TIMEZONE)
        expected_hours = int((next_day - day) / pd.Timedelta("1h"))
        complete = len(day_prices) == expected_hours and day_prices.notna().all()
        values = day_prices.to_numpy()
        rows[day.date()] = {
            f"savings_{n}h": one_cycle_savings(values, n) if complete else np.nan
            for n in BATTERY_HOURS
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "date"
    return df


def main():
    prices = read_da_prices()
    df = build_daily_savings(prices)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "da_arbitrage_savings_daily.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")
    print(df.describe().loc[["count", "mean", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
