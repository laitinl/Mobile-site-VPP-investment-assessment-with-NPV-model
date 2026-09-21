import numpy as np
import pandas as pd

from src.pipelines.reserve_prices import PROCESSED_DIR, build_reserve_prices

# Markets each battery can participate in, in tie-break order
BATTERY_MARKETS = {
    "1h": ["ffr", "fcr_d"],
    "3h": ["ffr", "fcr_d", "afrr"],
}
NO_MARKET = "none"


def build_market_revenues(prices: pd.DataFrame) -> pd.DataFrame:
    """Hourly revenue (€/MW) of participating in each market.

    FFR is procured only for up regulation. FCR-D and aFRR are bid in both
    directions, so their revenue is the sum of the up and down prices. A market is
    NaN in hours where any of its prices is missing (market not yet in operation).
    """
    return pd.DataFrame(
        {
            "ffr": prices["ffr_up"],
            "fcr_d": prices["fcr_d_up"] + prices["fcr_d_down"],
            "afrr": prices["afrr_up"] + prices["afrr_down"],
        }
    )


def select_best_market(revenues: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    """Best market among `markets` and its revenue for every hour.

    Only one market can be chosen per hour. NaN markets are not available. Hours
    where no market pays anything get "none". Hours where no market has data get NaN.
    """
    revenues = revenues[markets]
    best_price = revenues.max(axis=1, skipna=True)
    best_market = pd.Series(np.nan, index=revenues.index, dtype=object)
    available = revenues.notna().any(axis=1)
    best_market[available] = revenues[available].idxmax(axis=1, skipna=True)
    best_market[available & (best_price <= 0)] = NO_MARKET
    return pd.DataFrame({"best_price": best_price, "best_market": best_market})


def build_best_market_plans(prices: pd.DataFrame) -> pd.DataFrame:
    """Market revenues and the best market plan of every battery case, by hour."""
    revenues = build_market_revenues(prices)
    plans = [
        select_best_market(revenues, markets).add_suffix(f"_{battery}")
        for battery, markets in BATTERY_MARKETS.items()
    ]
    return pd.concat([revenues, *plans], axis=1)


def main():
    prices = build_reserve_prices()
    df = build_best_market_plans(prices)
    df.index.name = "start_time_utc"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "best_reserve_market_hourly.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")
    for battery in BATTERY_MARKETS:
        print(f"Best market of the {battery} battery:")
        print(df[f"best_market_{battery}"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
