import unittest
import numpy as np
import pandas as pd
from src.pipelines.daily_operation import (
    build_daily_reserve_yields,
    select_operation,
)


class TestSelectOperation(unittest.TestCase):
    def setUp(self):
        self.reserve = pd.DataFrame(
            {"ffr": [30.0, 10.0, 10.0, np.nan, 5.0], "fcr_d": [70.0, 40.0, 40.0, np.nan, 5.0]}
        )
        self.da = pd.Series([60.0, 60.0, 50.0, 40.0, np.nan])
        self.operation = select_operation(self.reserve, self.da)

    def test_reserve_only_when_yield_exceeds_da_savings(self):
        # Reserve yields are 100, 50, 50, missing; DA savings 60, 60, 50, 40
        self.assertEqual(self.operation["mode"][:4].tolist(), ["reserve", "da", "da", "da"])

    def test_reserve_days_report_yield_of_each_market(self):
        self.assertEqual(self.operation.loc[0, ["ffr", "fcr_d", "da"]].tolist(), [30, 70, 0])

    def test_da_days_report_arbitrage_savings(self):
        self.assertEqual(self.operation.loc[1, ["ffr", "fcr_d", "da"]].tolist(), [0, 0, 60])
        self.assertEqual(self.operation.loc[3, ["ffr", "fcr_d", "da"]].tolist(), [0, 0, 40])

    def test_missing_da_savings_gives_nan(self):
        self.assertTrue(pd.isna(self.operation.loc[4, "mode"]))
        self.assertTrue(self.operation.loc[4, ["ffr", "fcr_d", "da"]].isna().all())


class TestDailyReserveYields(unittest.TestCase):
    def setUp(self):
        # 2024-01-01 22:00 UTC is 2024-01-02 00:00 in Finland (UTC+2)
        index = pd.date_range("2024-01-01 22:00", periods=30, freq="h", tz="UTC")
        self.plans = pd.DataFrame(
            {
                "best_price_1h": 2.0,
                "best_market_1h": "fcr_d",
                "best_price_3h": 3.0,
                "best_market_3h": "afrr",
            },
            index=index,
        )
        self.plans.iloc[:4, 2:] = [[9.0, "ffr"]] * 4

    def test_yield_goes_to_the_market_of_each_hour(self):
        daily = build_daily_reserve_yields(self.plans, "3h")
        self.assertEqual(daily.index[0], pd.Timestamp("2024-01-02"))
        # 4 hours of FFR at 9 and 20 hours of aFRR at 3
        self.assertEqual(daily.loc["2024-01-02", ["ffr", "fcr_d", "afrr"]].tolist(), [36, 0, 60])
        self.assertEqual(daily.columns.tolist(), ["ffr", "fcr_d", "afrr"])

    def test_1h_battery_has_no_afrr(self):
        daily = build_daily_reserve_yields(self.plans, "1h")
        self.assertEqual(daily.columns.tolist(), ["ffr", "fcr_d"])
        self.assertEqual(daily.loc["2024-01-02", "fcr_d"], 48)

    def test_incomplete_days_are_nan(self):
        daily = build_daily_reserve_yields(self.plans, "3h")
        # The second day only has 6 of its 24 hours
        self.assertTrue(daily.iloc[1].isna().all())
        self.plans.iloc[10, 0] = np.nan
        daily = build_daily_reserve_yields(self.plans, "1h")
        self.assertTrue(daily.iloc[0].isna().all())


if __name__ == "__main__":
    unittest.main()
