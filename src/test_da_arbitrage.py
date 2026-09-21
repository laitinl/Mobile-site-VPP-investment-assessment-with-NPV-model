import unittest
import numpy as np
from src.pipelines.da_arbitrage import one_cycle_savings


class TestOneCycleSavings(unittest.TestCase):
    def test_1h_battery_buys_low_before_selling_high(self):
        prices = np.array([50, 10, 80, 20, 60], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 1), 70)

    def test_no_savings_when_prices_only_fall(self):
        prices = np.array([90, 70, 50, 30], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 1), 0)
        self.assertEqual(one_cycle_savings(prices, 3), 0)

    def test_3h_battery_uses_three_cheapest_and_dearest_hours(self):
        prices = np.array([1, 2, 3, 10, 20, 30, 4, 5], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 3), (10 + 20 + 30) - (1 + 2 + 3))

    def test_discharge_cannot_precede_charge(self):
        # Cheap hours only come after the expensive ones
        prices = np.array([10, 10, 10, 1, 1, 1], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 3), 0)

    def test_interleaved_charge_and_discharge(self):
        # Charge 2 cheap hours, sell one, buy one more cheap, sell the rest
        prices = np.array([1, 1, 9, 1, 9, 9], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 3), 24)

    def test_partial_cycle_when_only_some_hours_are_profitable(self):
        prices = np.array([10, 50, 20, 20, 20, 20], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 3), 40)

    def test_negative_prices_are_paid_to_charge(self):
        prices = np.array([-20, 30], dtype=float)
        self.assertEqual(one_cycle_savings(prices, 1), 50)


if __name__ == "__main__":
    unittest.main()
