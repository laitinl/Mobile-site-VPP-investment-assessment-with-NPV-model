import unittest
import numpy as np
from npv_simulator import NPVSimulator


class TestNPVSimulator(unittest.TestCase):
    def setUp(self):
        self.cases = {
            "Investment_size": np.array([1, 4, 1, 4, 1, 4]),  # Extra capacity hours
            "FCR weight": np.array([0, 0, 1, 0.5, 1, 0.5]),  # Weight of FCR revenues
            "aFRR weight": np.array([0, 0, 0, 0.7, 0, 0.7]),  # Weight of aFRR revenues
            "LS weight": np.array(
                [1, 1, 0, 0, 0, 0]
            ),  # Weight of load shifting revenues
            "Controller": np.array(
                [False, False, False, False, True, True]
            ),  # Whether a VPP controller is used
        }
        self.config = {
            "n_years": 10,
            "battery_capacity_cost": 200,  # Cost per kWh
            "battery_installation_cost": 1000,  # Installation cost per site
            "vpp_controller_cost": 1000,  # Cost for VPP controller per
            "fcr_yield": 108000,  # Yield from FCR-D up market €/MW/year
            "afrr_yield": 204000,  # Yield from aFRR market €/MW/year
            "load_shifting_savings": (30 * 2 + 15 * 2)
            * 365,  # Savings from load shifting €/MW/year
            "bsp_fee_min": 0.10,  # BSP share of revenue
            "bsp_fee_max": 0.25,
            "site_mean_power": 4.5,  # Mean power per site in kW
            "vpp_total_power": 1000,  # Total power of the VPP in kW
            "discount_rate": 0.08,  # Discount rate for NPV calculation
            "reserve_price_multiplier_min": -0.5,  # Min multiplier for reserve market
            "reserve_price_multiplier_max": 0.2,  # Max multiplier for reserve market
            "spot_price_multiplier_min": -0.2,  # Min multiplier for spot price
            "spot_price_multiplier_max": 0.2,  # Max multiplier for spot price
            "tax_rate": 0.2,  # Tax rate for NPV calculation
        }

    def test_run(self):
        simulator = NPVSimulator(self.cases, self.config)
        results = simulator.run(count=100)
        self.assertEqual(results.shape[2], len(self.cases["Investment_size"]))

    def test_npv_scenario_5(self):
        simulator = NPVSimulator(self.cases, self.config)
        results = simulator.run(count=1000, random_seed=42)

        np.random.seed(42)

        # Fixed parameters
        n_years = self.config["n_years"]
        battery_capacity_cost = self.config["battery_capacity_cost"]
        battery_installation_cost = self.config["battery_installation_cost"]
        vpp_controller_cost = self.config["vpp_controller_cost"]
        reserve_market_yield = self.config["fcr_yield"]
        load_shifting_savings = 0 * self.config["load_shifting_savings"]
        site_mean_power = self.config["site_mean_power"]
        vpp_total_power = self.config["vpp_total_power"]
        discount_rate = self.config["discount_rate"]
        tax_rate = self.config["tax_rate"]

        # Derived parameters
        n_sites = vpp_total_power // site_mean_power
        battery_capacity = vpp_total_power  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites * battery_installation_cost
            + n_sites * vpp_controller_cost
        )
        deprecation = investment_cost / n_years

        # Sampled parameters
        reserve_price_multiplier = np.random.uniform(
            self.config["reserve_price_multiplier_min"],
            self.config["reserve_price_multiplier_max"],
            1000,
        )
        spot_price_multiplier = np.random.uniform(
            self.config["spot_price_multiplier_min"],
            self.config["spot_price_multiplier_max"],
            1000,
        )
        bsp_fee = np.random.uniform(
            self.config["bsp_fee_min"],
            self.config["bsp_fee_max"],
            1000,
        )

        # Calculate cash flows
        cash_flows = np.zeros((1000, n_years + 1))
        cash_flows[:, 0] = -investment_cost

        for year in range(1, n_years + 1):
            reserve_market_revenue = (
                reserve_market_yield
                * (1 + reserve_price_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            load_shifting_revenue = (
                load_shifting_savings
                * (1 + spot_price_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )

            total_revenue = reserve_market_revenue + load_shifting_revenue
            ebit = total_revenue * (1 - bsp_fee) - deprecation

            net_income = ebit
            net_income[net_income > 0] -= tax_rate * net_income[net_income > 0]
            cash_flows[:, year] = net_income + deprecation

        years = np.arange(0, n_years + 1)
        npv = np.cumsum(cash_flows / ((1 + discount_rate) ** years), axis=1)

        self.assertAlmostEqual(np.linalg.norm(npv - results[:, :, 4]), 0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
