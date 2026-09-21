from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Fingrid data IDs, see documentation/raw_data_ids.md
RESERVE_MARKETS = {
    277: "ffr_up",
    318: "fcr_d_up",
    283: "fcr_d_down",
    52: "afrr_up",
    51: "afrr_down",
}
AFRR_COLUMNS = ["afrr_up", "afrr_down"]
AFRR_DISABLED_HOURS = 4  # Gaps of exactly this length are hours when aFRR was disabled


def read_raw_series(data_id: int, name: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Read one raw Fingrid price file as an hourly series indexed by UTC start time."""
    df = pd.read_csv(
        raw_dir / f"raw_{data_id}.csv",
        usecols=["start_time_utc", "end_time_utc", "value"],
        parse_dates=["start_time_utc", "end_time_utc"],
    )
    if not (df["end_time_utc"] - df["start_time_utc"] == pd.Timedelta("1h")).all():
        raise ValueError(f"raw_{data_id}.csv contains non-hourly rows")
    if df["start_time_utc"].duplicated().any():
        raise ValueError(f"raw_{data_id}.csv contains duplicate timestamps")
    return df.set_index("start_time_utc")["value"].rename(name).sort_index()


def fill_gaps_with_zero(s: pd.Series, gap_length: int) -> pd.Series:
    """Set runs of exactly `gap_length` consecutive NaNs to 0."""
    is_na = s.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")
    return s.mask(is_na & (run_length == gap_length), 0.0)


def build_reserve_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Match all reserve market prices (€/MW) by hour.

    The index spans every hour from the earliest to the latest timestamp in any
    market. Gaps of exactly 4 hours in the aFRR series are hours when aFRR was
    disabled and are set to 0. All other gaps within a market's own data range are
    linearly interpolated. Hours before a market's first observation stay NaN, and
    reported zeros are kept as is.
    """
    series = [read_raw_series(i, name, raw_dir) for i, name in RESERVE_MARKETS.items()]
    df = pd.concat(series, axis=1)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "start_time_utc"
    for col in AFRR_COLUMNS:
        df[col] = fill_gaps_with_zero(df[col], AFRR_DISABLED_HOURS)
    return df.interpolate(method="linear", limit_area="inside")


def main():
    df = build_reserve_prices()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "reserve_prices_hourly.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")
    print(df.describe().loc[["count", "mean", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
