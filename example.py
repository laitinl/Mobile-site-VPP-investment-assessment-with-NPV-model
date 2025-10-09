# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.npv_simulator import NPVSimulator
from src.utils.plotting import (
    plot_cumulative_npv,
    plot_npv_hist_grid,
    plot_npv_boxplots,
    plot_npv_hist_overlaid,
)


def main():
    cases = {
        "Investment_size": np.array([1, 4, 1, 4, 1, 4]),  # Extra capacity hours
        "FCR weight": np.array([0, 0, 1, 0, 1, 0]),  # Weight of FCR revenues
        "aFRR weight": np.array([0, 0, 0, 1, 0, 1]),  # Weight of aFRR revenues
        "LS weight": np.array(
            [0.25, 1, 0, 0, 0, 0]
        ),  # Weight of load shifting revenues
        "Controller": np.array(
            [False, False, False, False, True, True]
        ),  # Whether a VPP controller is used
    }

    config = {
        "n_years": 10,
        "battery_capacity_cost": 100,  # Cost per kWh
        "battery_installation_cost": (
            500,
            25,
        ),  # Installation cost per site (fixed cost, cost per kWh)
        "vpp_controller_cost": 1500,  # Cost for VPP controller per
        "fcr_yield": 109000 + 118000,  # Yield from FCR-D up and down market €/MW/year
        "afrr_yield": 176000 + 141000,  # Yield from aFRR up and down market €/MW/year
        "load_shifting_savings": (40 * 4) * 365,  # Savings from load shifting €/MW/year
        "peak_shaving_savings_per_site": (1.35 * 2)
        * 12,  # Single site savings from peak shaving €/MW/year
        "connectivity_cost": 240,  # VPP connectivity cost per year
        "o&m_cost": 0.02,  # O&M cost as a fraction of investment cost
        "bsp_fee_dist": (
            0.1,
            0.2,
            0.3,
        ),  # BSP fee distribution parameters (min, mode, max)
        "site_mean_power": 2,  # Mean power per site in kW
        "vpp_total_power": 1000,  # Total power of the VPP in kW
        "discount_rate": 0.057,  # Discount rate for NPV calculation
        "reserve_price_multiplier_dist": (
            -0.5,
            -0.2,
            0.5,
        ),  # Reserve price multiplier distribution (min, mode, max)
        "spot_volatility_multiplier_dist": (
            -0.1,
            0.0,
            0.1,
        ),  # Spot price multiplier distribution (min, mode, max)
        "power_charge_multiplier_dist": (
            -0.1,
            0.2,
            0.96,
        ),  # Power charge multiplier distribution (min, mode, max)
    }

    def make_tornado_plot(sens_params, sens_array, scenario, title=None):
        df_sensitivity = pd.DataFrame(
            {
                "param_names": sens_params,
                "sens_plus": sens_array[:, 0, scenario],
                "sens_minus": sens_array[:, 1, scenario],
            }
        )
        df_sensitivity["delta"] = np.abs(
            df_sensitivity["sens_plus"] - df_sensitivity["sens_minus"]
        )
        df_sensitivity.sort_values("delta", inplace=True)
        plt.grid(axis="y", alpha=0.5)
        plt.barh(
            np.arange(len(df_sensitivity)),
            df_sensitivity["sens_plus"],
            label="Parameter increase 20%",
        )
        plt.barh(
            np.arange(len(df_sensitivity)),
            df_sensitivity["sens_minus"],
            label="Parameter decrease 20%",
        )
        plt.yticks(np.arange(len(df_sensitivity)), df_sensitivity["param_names"])
        plt.legend()
        plt.title(title)
        plt.xlabel("NPV change (€)")
        plt.show()

    simulator = NPVSimulator(cases, config)
    sens_params, sens_array = simulator.run_sensitivity_analysis()
    scenarios = ["1a", "1b", "2a", "2b", "3a", "3b"]
    for i in range(6):
        make_tornado_plot(sens_params, sens_array, i, title=f"Scenario {scenarios[i]}")

    results_with_different_site_power = []
    results = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)
    scenario = 2
    npv_last_year = results[:, -1, scenario]
    print(f"Median NPV: {np.median(npv_last_year)}")
    # print(f"Standard Deviation of NPV: {np.std(npv_last_year)}")
    print(f"Minimum NPV: {np.min(npv_last_year)}")
    print(f"Maximum NPV: {np.max(npv_last_year)}")
    print(f"5th Percentile NPV: {np.percentile(npv_last_year, 5)}")
    print(f"95th Percentile NPV: {np.percentile(npv_last_year, 95)}")

    plt.figure(figsize=(10, 6))
    plt.hist(npv_last_year, bins=500, density=True)
    plt.xlabel("NPV (€)")
    plt.ylabel("Density")
    plt.show()

    plot_cumulative_npv(results, scenario, config["n_years"])

    # Boxplot for different site mean power scenarios
    config["site_mean_power"] = 5
    simulator = NPVSimulator(cases, config)
    results = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    config["site_mean_power"] = 10
    simulator = NPVSimulator(cases, config)
    results = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    plot_npv_boxplots(
        site_powers=[2, 5, 10],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )

    plot_npv_hist_grid(
        site_powers=[2, 5, 10],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )

    plot_npv_hist_overlaid(
        site_powers=[2, 5, 10],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )


if __name__ == "__main__":
    main()

# %%
