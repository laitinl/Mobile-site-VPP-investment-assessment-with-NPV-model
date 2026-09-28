import unittest
import numpy as np
from src.npv_simulator import NPVSimulator


class TestNPVSimulator(unittest.TestCase):
    def setUp(self):
        eps = 1e-12  # Small epsilon
        self.cases = {
            "Investment_size": np.array(
                [0, 1, 3, 0, 1, 3, 0, 1, 3]
            ),  # Extra capacity hours
            "FFR yield": np.array(
                [0, 0, 0, 16960, 16960, 16960, 16960, 16960, 16960]
            ),  # Yield from FFR up market €/MW/year
            "FCR-D yield": np.array(
                [0, 0, 0, 0, 67960, 0.44 * 94710, 0, 67960, 0.44 * 94710]
            ),  # Yield from FCR-D up and downmarket €/MW/year
            "aFRR yield": np.array(
                [0, 0, 0, 0, 0, 0.56 * 94710, 0, 0, 0.56 * 94710]
            ),  # Yield from aFRR up and downmarket €/MW/year
            "LS savings": np.array(
                [0, 21330, 60070, 0, 21330, 60070, 0, 21330, 60070]
            ),  # Savings from load shifting €/MW/year
            "Controller": np.array(
                [False, False, False, False, False, False, True, True, True]
            ),  # Whether a VPP controller is used
            "Connectivity": np.array(
                [False, True, True, True, True, True, True, True, True]
            ),  # Whether VPP connectivity is used
            "Peak shaving": np.array(
                [False, True, True, False, True, True, False, True, True]
            ),  # Whether peak shaving is used
        }

        self.config = {
            "n_years": 10,
            "battery_capacity_cost": (115 - eps, 115, 115 + eps),  # Cost per kWh
            "battery_installation_cost": (
                500,
                25,
            ),  # Installation cost per site (fixed cost, variable cost per kWh)
            "vpp_controller_cost": (
                500 - eps,
                500,
                500 + eps,
            ),  # Cost for VPP controller per
            "ffr_weight": 1,  # Weight of FFR revenues
            "fcr_weight": 1,  # Weight of FCR revenues
            "afrr_weight": 1,  # Weight of aFRR revenues
            "ls_weight": 1,  # Weight of load shifting revenues
            "peak_shaving_savings_per_site": 0,  # Single site savings from peak shaving €/MW/year
            "connectivity_cost": (
                12 - eps,
                12,
                12 + eps,
            ),  # VPP connectivity cost per year
            "o&m_cost": (
                0.02 - eps,
                0.02,
                0.02 + eps,
            ),  # O&M cost as a fraction of investment cost
            "bsp_fee_dist": (
                0.20 - eps,
                0.20,
                0.20 + eps,
            ),  # BSP fee distribution parameters (min, mode, max)
            "site_mean_power": 2.5,  # Mean power per site in kW
            "vpp_total_power": 1000,  # Total power of the VPP in kW
            "discount_rate": (
                0.057 - eps,
                0.057,
                0.057 + eps,
            ),  # Discount rate for NPV calculation
            "reserve_price_multiplier_dist": (
                -0.2 - eps,
                -0.2,
                -0.2 + eps,
            ),  # Reserve price multiplier distribution (min, mode, max)
            "spot_volatility_multiplier_dist": (
                0.0 - eps,
                0.0,
                0.0 + eps,
            ),  # Spot price multiplier distribution (min, mode, max)
        }

    def test_run(self):
        simulator = NPVSimulator(self.cases, self.config)
        results, _, _ = simulator.run_uncertainty_analysis(count=100)
        self.assertEqual(results.shape[2], len(self.cases["Investment_size"]))

    def test_npv_scenario_5(self):
        scenario_index = 4  # Index for the scenario with 1h battery, FFR, FCR-D, aFRR, load shifting, controller, connectivity, and peak shaving
        simulator = NPVSimulator(self.cases, self.config)
        results, _, _ = simulator.run_uncertainty_analysis(count=100, random_seed=42)

        np.random.seed(42)

        # Fixed parameters
        n_years = self.config["n_years"]
        battery_capacity_cost = self.config["battery_capacity_cost"][1]
        fixed_battery_installation_cost, variable_battery_installation_cost = (
            self.config["battery_installation_cost"]
        )
        vpp_controller_cost = self.config["vpp_controller_cost"][1]
        site_mean_power = self.config["site_mean_power"]
        vpp_total_power = self.config["vpp_total_power"]
        discount_rate = self.config["discount_rate"][1]
        connectivity_cost = self.config["connectivity_cost"][1]
        om_cost = self.config["o&m_cost"][1]

        # Derived parameters
        n_sites = np.ceil(vpp_total_power / site_mean_power).astype(int)
        battery_capacity = (
            self.cases["Investment_size"][scenario_index] * vpp_total_power
        )  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites
            * fixed_battery_installation_cost
            * np.abs(np.sign(self.cases["Investment_size"][scenario_index]))
            + battery_capacity * variable_battery_installation_cost
            + n_sites * vpp_controller_cost * self.cases["Controller"][scenario_index]
        )
        om_expense = (
            battery_capacity * battery_capacity_cost
            + n_sites * vpp_controller_cost * self.cases["Controller"][scenario_index]
        ) * om_cost
        annual_cost = (
            om_expense
            + connectivity_cost * n_sites * self.cases["Connectivity"][scenario_index]
        )
        reserve_market_yield = (
            self.cases["FFR yield"][scenario_index]
            + self.cases["FCR-D yield"][scenario_index]
            + self.cases["aFRR yield"][scenario_index]
        )

        # Sampled parameters
        reserve_price_multiplier = self.config["reserve_price_multiplier_dist"][1]
        spot_price_multiplier = self.config["spot_volatility_multiplier_dist"][1]
        bsp_fee = self.config["bsp_fee_dist"][1]

        # Calculate cash flows
        cash_flows = np.zeros((100, n_years + 1))
        cash_flows[:, 0] = -investment_cost

        for year in range(1, n_years + 1):
            reserve_market_revenue = (
                reserve_market_yield
                * (1 + reserve_price_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            load_shifting_revenue = (
                self.cases["LS savings"][scenario_index]
                * (1 + spot_price_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )

            total_revenue = (
                reserve_market_revenue * (1 - bsp_fee) + load_shifting_revenue
            )
            cash_flows[:, year] = total_revenue - annual_cost

        years = np.arange(0, n_years + 1)
        npv = np.cumsum(cash_flows / ((1 + discount_rate) ** years), axis=1)

        self.assertAlmostEqual(
            np.linalg.norm(npv - results[:, :, scenario_index]), 0, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
