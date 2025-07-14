# %%
import numpy as np
import matplotlib.pyplot as plt


class NPVSimulator:
    """
    A class to simulate Net Present Value (NPV) for different mobile site
    virtual power plant (VPP) battery investment scenarios.
    """

    def __init__(self, cases: dict, config: dict):
        """
        Initialize the NPV simulator with a configuration dictionary.

        Args:
        cases (dict): Dictionary containing different investment scenarios.
                      Keys are scenario names and values are lists of parameters.
        config (dict): Configuration parameters for the simulation.
        """
        self.cases = cases
        self.config = config

    def calculate_npv(self, cash_flows: np.ndarray, discount_rate: float) -> np.ndarray:
        """
        Calculate the Net Present Value (NPV) for given cash flows and discount rate.

        Args:
        cash_flows (np.ndarray): Array of cash flows.
        discount_rate (float): Discount rate for NPV calculation.

        Returns:
        np.ndarray: NPV values.
        """
        years = np.arange(0, self.config["n_years"] + 1)
        cash_flows = (
            cash_flows * (1 + discount_rate) ** -years[np.newaxis, :, np.newaxis]
        )
        npvs = np.cumsum(cash_flows, axis=1)
        return npvs

    def run(self, count: int, random_seed: int = 42):
        """
        Run the NPV simulation.

        Args:
        count (int): Number of simulations to run.
        random_state (int): Seed for random number generation.
        """
        np.random.seed(random_seed)

        # Fixed parameters
        n_years = self.config["n_years"]
        battery_capacity_cost = self.config["battery_capacity_cost"]
        battery_installation_cost = self.config["battery_installation_cost"]
        vpp_controller_cost = self.config["vpp_controller_cost"]
        fcr_yield = self.config["fcr_yield"]
        afrr_yield = self.config["afrr_yield"]
        load_shifting_savings = self.config["load_shifting_savings"]
        site_mean_power = self.config["site_mean_power"]
        vpp_total_power = self.config["vpp_total_power"]
        discount_rate = self.config["discount_rate"]
        tax_rate = self.config["tax_rate"]

        # Derived parameters
        n_sites = vpp_total_power // site_mean_power
        battery_capacity = (
            self.cases["Investment_size"] * vpp_total_power
        )  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites * battery_installation_cost
            + n_sites * vpp_controller_cost * self.cases["Controller"]
        )
        deprecation = investment_cost / n_years
        reserve_market_yield = (
            self.cases["FCR weight"] * fcr_yield
            + self.cases["aFRR weight"] * afrr_yield
        )

        # Sampled parameters
        reserve_price_multiplier = np.random.uniform(
            self.config["reserve_price_multiplier_min"],
            self.config["reserve_price_multiplier_max"],
            count,
        )
        spot_price_multiplier = np.random.uniform(
            self.config["spot_price_multiplier_min"],
            self.config["spot_price_multiplier_max"],
            count,
        )
        bsp_fee = np.random.uniform(
            self.config["bsp_fee_min"],
            self.config["bsp_fee_max"],
            count,
        )

        # Calculate cash flows
        cash_flows = np.zeros(
            (count, n_years + 1, self.cases["Investment_size"].shape[0])
        )
        cash_flows[:, 0, :] = -investment_cost

        for year in range(1, n_years + 1):
            reserve_market_revenue = (
                reserve_market_yield[np.newaxis, :]
                * (1 + reserve_price_multiplier[:, np.newaxis] * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )
            load_shifting_revenue = (
                self.cases["LS weight"][np.newaxis, :]
                * load_shifting_savings
                * (1 + spot_price_multiplier[:, np.newaxis] * (year + 1) / n_years)
                * vpp_total_power
                / 1000
            )

            total_revenue = reserve_market_revenue + load_shifting_revenue
            ebit = (
                total_revenue * (1 - bsp_fee[:, np.newaxis])
                - deprecation[np.newaxis, :]
            )

            net_income = ebit
            net_income[net_income > 0] -= tax_rate * net_income[net_income > 0]
            cash_flows[:, year, :] = net_income + deprecation[np.newaxis, :]

        # Calculate NPV
        out = self.calculate_npv(cash_flows, discount_rate)

        return out


# %%
def main():
    cases = {
        "Investment_size": np.array([1, 4, 1, 4, 1, 4]),  # Extra capacity hours
        "FCR weight": np.array([0, 0, 1, 0.5, 1, 0.5]),  # Weight of FCR revenues
        "aFRR weight": np.array([0, 0, 0, 0.7, 0, 0.7]),  # Weight of aFRR revenues
        "LS weight": np.array(
            [0.5, 1, 0.5, 1, 0.5, 1]
        ),  # Weight of load shifting revenues
        "Controller": np.array(
            [False, False, False, False, True, True]
        ),  # Whether a VPP controller is used
    }

    config = {
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

    simulator = NPVSimulator(cases, config)
    results = simulator.run(count=3000000)
    scenario = 3
    npv_last_year = results[:, -1, scenario]
    print(f"Mean NPV: {np.mean(npv_last_year)}")
    print(f"Standard Deviation of NPV: {np.std(npv_last_year)}")
    print(f"Minimum NPV: {np.min(npv_last_year)}")
    print(f"Maximum NPV: {np.max(npv_last_year)}")
    print(f"5th Percentile NPV: {np.percentile(npv_last_year, 5)}")
    print(f"95th Percentile NPV: {np.percentile(npv_last_year, 95)}")

    plt.figure(figsize=(10, 6))
    plt.hist(npv_last_year, bins=500, density=True)
    plt.xlabel("NPV (€)")
    plt.ylabel("Density")
    plt.show()

    percentiles_npv = np.percentile(results[:, :, scenario], [2.5, 50, 97.5], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(0, config["n_years"] + 1),
        percentiles_npv[1],
        label="Median",
        color="blue",
    )
    plt.plot(
        np.arange(0, config["n_years"] + 1),
        percentiles_npv[::2, :].T,
        "--",
        label="Confidence Interval 95%",
        color="blue",
    )
    plt.xlabel("Year")
    plt.ylabel("NPV (€)")
    plt.grid(axis="y", alpha=0.5)
    plt.legend()
    plt.show()

    # Boxplot for different site mean power scenarios
    positions = np.arange(1, len(cases["Investment_size"]) * 3, 3)
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        results[:, -1, :],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="blue"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        positions=positions + 0.2,
        label="4.5 kW Site Mean Power",
    )

    config["site_mean_power"] = 8
    simulator = NPVSimulator(cases, config)
    results = simulator.run(count=3000000)
    plt.boxplot(
        results[:, -1, :],
        patch_artist=True,
        boxprops=dict(facecolor="plum", color="purple"),
        medianprops=dict(color="purple"),
        whiskerprops=dict(color="purple"),
        capprops=dict(color="purple"),
        positions=positions + 1,
        label="8 kW Site Mean Power",
    )

    config["site_mean_power"] = 2.5
    simulator = NPVSimulator(cases, config)
    results = simulator.run(count=3000000)
    plt.boxplot(
        results[:, -1, :],
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="green"),
        whiskerprops=dict(color="green"),
        capprops=dict(color="green"),
        positions=positions + 1.8,
        label="2.5 kW Site Mean Power",
    )
    plt.xticks(
        positions + 1,
        [f"Scenario {i + 1}" for i in range(len(cases["Investment_size"]))],
    )
    plt.ylabel("NPV (€)")
    plt.grid(axis="y", alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
