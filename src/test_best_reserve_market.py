import unittest
import numpy as np
import pandas as pd
from src.pipelines.best_reserve_market import (
    build_best_market_plans,
    build_market_revenues,
    select_best_market,
)


class TestBestReserveMarket(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame(
            {
                "ffr_up": [50, 10, 0, np.nan, np.nan],
                "fcr_d_up": [10, 10, 0, 5, np.nan],
                "fcr_d_down": [10, 10, 0, np.nan, np.nan],
                "afrr_up": [15, 30, 0, 20, np.nan],
                "afrr_down": [15, 1, 0, 1, np.nan],
            }
        )

    def test_fcr_d_and_afrr_are_paid_for_both_directions(self):
        revenues = build_market_revenues(self.prices)
        self.assertEqual(revenues["fcr_d"][0], 20)
        self.assertEqual(revenues["afrr"][1], 31)

    def test_best_market_and_price(self):
        revenues = build_market_revenues(self.prices)
        df = select_best_market(revenues, ["ffr", "fcr_d", "afrr"])
        self.assertEqual(df["best_market"][0], "ffr")
        self.assertEqual(df["best_price"][0], 50)
        self.assertEqual(df["best_market"][1], "afrr")
        self.assertEqual(df["best_price"][1], 31)

    def test_no_market_pays(self):
        revenues = build_market_revenues(self.prices)
        df = select_best_market(revenues, ["ffr", "fcr_d", "afrr"])
        self.assertEqual(df["best_market"][2], "none")
        self.assertEqual(df["best_price"][2], 0)

    def test_unavailable_markets_are_skipped(self):
        revenues = build_market_revenues(self.prices)
        df = select_best_market(revenues, ["ffr", "fcr_d", "afrr"])
        self.assertEqual(df["best_market"][3], "afrr")
        self.assertEqual(df["best_price"][3], 21)

    def test_no_data_gives_nan(self):
        revenues = build_market_revenues(self.prices)
        df = select_best_market(revenues, ["ffr", "fcr_d", "afrr"])
        self.assertTrue(np.isnan(df["best_price"][4]))
        self.assertTrue(pd.isna(df["best_market"][4]))

    def test_1h_battery_cannot_use_afrr(self):
        df = build_best_market_plans(self.prices)
        # Hour 1: aFRR pays 31 but is not available to the 1h battery
        self.assertEqual(df["best_market_1h"][1], "fcr_d")
        self.assertEqual(df["best_price_1h"][1], 20)
        self.assertEqual(df["best_market_3h"][1], "afrr")
        self.assertEqual(df["best_price_3h"][1], 31)
        # Hour 3: only aFRR has data, so the 1h battery has no market
        self.assertTrue(pd.isna(df["best_market_1h"][3]))
        self.assertEqual(df["best_market_3h"][3], "afrr")


if __name__ == "__main__":
    unittest.main()
