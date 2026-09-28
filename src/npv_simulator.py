# %%
import numpy as np
import pandas as pd
from betapert import pert


class NPVSimulator:
    """
    A class to simulate Net Present Value (NPV) for different mobile site
    virtual power plant (VPP) battery investment scenarios.
    """

    def __init__(self, cases: dict[str, np.ndarray], config: dict):
        """
        Initialize the NPV simulator with a configuration dictionary.

        Args:
        cases (dict): Dictionary containing different investment scenarios.
                      Keys are scenario names and values are lists of parameters.
        config (dict): Configuration parameters for the simulation.
        """
        self.cases = cases
        self.config = config

    def _calculate_npv(
        self, cash_flows: np.ndarray, discount_rate: np.array
    ) -> np.ndarray:
        """
        Calculate the Net Present Value (NPV) for given cash flows and discount rate.

        Args:
        cash_flows (np.ndarray): Array of cash flows.
        discount_rate (np.array): Discount rates for NPV calculation.

        Returns:
        np.ndarray: NPV values.
        """
        years = np.arange(0, self.config["n_years"] + 1)
        cash_flows = (
            cash_flows
            * (1 + discount_rate[:, np.newaxis, np.newaxis])
            ** -years[np.newaxis, :, np.newaxis]
        )
        npvs = np.cumsum(cash_flows, axis=1)
        return npvs

    def run_uncertainty_analysis(self, count: int, random_seed: int = 42) -> np.ndarray:
        """
        Run the NPV simulation.

        Args:
        count (int): Number of simulations to run.
        random_state (int): Seed for random number generation.

        Returns:
        np.ndarray: Array of NPV results for each scenario and year.
        """
        np.random.seed(random_seed)

        # Constant parameters
        n_years = self.config["n_years"]
        fixed_battery_installation_cost, variable_battery_installation_cost = (
            self.config["battery_installation_cost"]
        )
        site_mean_power = self.config["site_mean_power"]
        vpp_total_power = self.config["vpp_total_power"]

        # Sampled parameters
        battery_capacity_cost = pert(*self.config["battery_capacity_cost"]).rvs(
            size=count
        )
        vpp_controller_cost = pert(*self.config["vpp_controller_cost"]).rvs(size=count)
        discount_rate = pert(*self.config["discount_rate"]).rvs(size=count)
        connectivity_cost = pert(*self.config["connectivity_cost"]).rvs(size=count)
        om_cost = pert(*self.config["o&m_cost"]).rvs(size=count)
        reserve_price_multiplier = pert(
            *self.config["reserve_price_multiplier_dist"]
        ).rvs(size=count)
        spot_volatility_multiplier = pert(
            *self.config["spot_volatility_multiplier_dist"]
        ).rvs(size=count)
        bsp_fee = pert(*self.config["bsp_fee_dist"]).rvs(size=count)

        # Derived parameters
        n_sites = np.ceil(vpp_total_power / site_mean_power).astype(int)
        battery_capacity = (
            self.cases["Investment_size"] * vpp_total_power
        )  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity[np.newaxis, :] * battery_capacity_cost[:, np.newaxis]
            + n_sites
            * fixed_battery_installation_cost
            * np.array([np.abs(np.sign(x)) for x in self.cases["Investment_size"]])[
                np.newaxis, :
            ]
            + battery_capacity[np.newaxis, :] * variable_battery_installation_cost
            + n_sites * vpp_controller_cost[:, np.newaxis] * self.cases["Controller"]
        )
        om_expense = (
            battery_capacity[np.newaxis, :] * battery_capacity_cost[:, np.newaxis]
            + n_sites * vpp_controller_cost[:, np.newaxis] * self.cases["Controller"]
        ) * om_cost[:, np.newaxis]
        annual_cost = (
            om_expense
            + connectivity_cost[:, np.newaxis]
            * n_sites
            * self.cases["Connectivity"][np.newaxis, :]
        )
        reserve_market_yield = (
            self.cases["FFR yield"]
            + self.cases["FCR-D yield"]
            + self.cases["aFRR yield"]
        )

        # Calculate cash flows
        cash_flows = np.zeros(
            (count, n_years + 1, self.cases["Investment_size"].shape[0])
        )
        revs = np.zeros((count, n_years + 1, self.cases["Investment_size"].shape[0]))
        cash_flows[:, 0, :] = -investment_cost

        for year in range(1, n_years + 1):
            reserve_market_revenue = (
                reserve_market_yield[np.newaxis, :]
                * (1 + reserve_price_multiplier[:, np.newaxis] * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            load_shifting_revenue = (
                self.cases["LS savings"][np.newaxis, :]
                * (1 + spot_volatility_multiplier[:, np.newaxis] * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            cash_flows[:, year, :] = (
                (reserve_market_revenue) * (1 - bsp_fee[:, np.newaxis])
                + load_shifting_revenue
                - annual_cost[np.newaxis, :]
            )
            revs[:, year, :] = reserve_market_revenue + load_shifting_revenue

        # Calculate NPV
        out = self._calculate_npv(cash_flows, discount_rate)

        return out, np.median(cash_flows, axis=0), np.median(revs, axis=0)

    def _calculate_npv_sensitivity(self, df):
        # Fixed parameters
        n_years = self.config["n_years"]
        battery_capacity_cost = df["battery_capacity_cost"].to_numpy()
        fixed_battery_installation_cost = df[
            "fixed_battery_installation_cost"
        ].to_numpy()
        variable_battery_installation_cost = df[
            "variable_battery_installation_cost"
        ].to_numpy()
        vpp_controller_cost = df["vpp_controller_cost"].to_numpy()
        ffr_weight = df["ffr_weight"].to_numpy()
        fcr_weight = df["fcr_weight"].to_numpy()
        afrr_weight = df["afrr_weight"].to_numpy()
        ls_weight = df["ls_weight"].to_numpy()
        site_mean_power = df["site_mean_power"].to_numpy()
        vpp_total_power = self.config["vpp_total_power"]
        discount_rate = df["discount_rate"].to_numpy()
        connectivity_cost = df["connectivity_cost"].to_numpy()
        bsp_fee = df["bsp_fee_dist"].to_numpy()
        reserve_price_multiplier = df["reserve_price_multiplier_dist"].to_numpy()
        spot_volatility_multiplier = df["spot_volatility_multiplier_dist"].to_numpy()
        power_charge_multiplier = df["power_charge_multiplier_dist"].to_numpy()

        # Derived parameters
        n_sites = vpp_total_power // site_mean_power
        battery_capacity = (
            self.cases["Investment_size"] * vpp_total_power
        )  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites
            * fixed_battery_installation_cost
            * np.array([[np.abs(np.sign(x)) for x in self.cases["Investment_size"]]])
            + battery_capacity * variable_battery_installation_cost
            + n_sites * vpp_controller_cost * self.cases["Controller"]
        )
        om_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites * vpp_controller_cost * self.cases["Controller"]
        ) * df["o&m_cost"].to_numpy()
        annual_cost = om_cost + connectivity_cost * n_sites * self.cases["Connectivity"]
        reserve_market_yield = (
            self.cases["FFR yield"] * ffr_weight
            + self.cases["FCR-D yield"] * fcr_weight
            + self.cases["aFRR yield"] * afrr_weight
        )

        # Calculate cash flows
        cash_flows = np.zeros((1, n_years + 1, self.cases["Investment_size"].shape[0]))
        cash_flows[:, 0, :] = -investment_cost

        for year in range(1, n_years + 1):
            reserve_market_revenue = (
                reserve_market_yield[np.newaxis, :]
                * (1 + reserve_price_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            load_shifting_revenue = (
                self.cases["LS savings"][np.newaxis, :]
                * ls_weight
                * (1 + spot_volatility_multiplier * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            cash_flows[:, year, :] = (
                reserve_market_revenue * (1 - bsp_fee)
                + load_shifting_revenue
                - annual_cost[np.newaxis, :]
            )

        # Calculate NPV
        out = self._calculate_npv(cash_flows, discount_rate)
        return out[0, -1, :]

    def run_sensitivity_analysis(self):
        param_names = [
            "battery_capacity_cost",
            "fixed_battery_installation_cost",
            "variable_battery_installation_cost",
            "vpp_controller_cost",
            "ffr_weight",
            "fcr_weight",
            "afrr_weight",
            "ls_weight",
            "peak_shaving_savings_per_site",
            "connectivity_cost",
            "o&m_cost",
            "bsp_fee_dist",
            "site_mean_power",
            "discount_rate",
            "reserve_price_multiplier_dist",
            "spot_volatility_multiplier_dist",
            "power_charge_multiplier_dist",
        ]
        param_values = [
            self.config["battery_capacity_cost"][1],
            self.config["battery_installation_cost"][0],
            self.config["battery_installation_cost"][1],
            self.config["vpp_controller_cost"][1],
            self.config["ffr_weight"],
            self.config["fcr_weight"],
            self.config["afrr_weight"],
            self.config["ls_weight"],
            self.config["peak_shaving_savings_per_site"],
            self.config["connectivity_cost"][1],
            self.config["o&m_cost"][1],
            self.config["bsp_fee_dist"][1],
            self.config["site_mean_power"],
            self.config["discount_rate"][1],
            self.config["reserve_price_multiplier_dist"][1],
            self.config["spot_volatility_multiplier_dist"][1],
            self.config["power_charge_multiplier_dist"][1],
        ]
        df_params = pd.DataFrame(np.array([param_values]), columns=param_names)
        base_npv = self._calculate_npv_sensitivity(df_params)
        sens_array = np.zeros(
            (len(param_names), 2, self.cases["Investment_size"].shape[0])
        )
        for i, param_name in enumerate(param_names):
            df_plus = df_params.copy()
            df_plus[param_name] *= 1.2
            df_minus = df_params.copy()
            df_minus[param_name] *= 0.8
            sens_array[i, 0, :] = self._calculate_npv_sensitivity(df_plus) - base_npv
            sens_array[i, 1, :] = self._calculate_npv_sensitivity(df_minus) - base_npv

        return param_names, sens_array
