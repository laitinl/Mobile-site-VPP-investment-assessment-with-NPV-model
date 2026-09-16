# %%
import numpy as np
import matplotlib.pyplot as plt
from src.npv_simulator import NPVSimulator
from src.utils.plotting import (
    plot_cumulative_npv,
    plot_npv_hist_grid,
    plot_npv_boxplots,
    plot_npv_hist_overlaid,
    plot_tornado,
    plot_tornado_grid,
)


def main():
    cases = {
        "Investment_size": np.array(
            [0, 1, 3, 0, 1, 3, 0, 1, 3]
        ),  # Extra capacity hours
        "FFR weight": np.array([0, 0, 0, 1, 0, 0, 1, 0, 0]),  # Weight of FFR revenues
        "FCR weight": np.array([0, 0, 0, 0, 1, 0, 0, 1, 0]),  # Weight of FCR revenues
        "aFRR weight": np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]),  # Weight of aFRR revenues
        "LS weight": np.array(
            [0, 0.33, 1, 0, 0.33, 1, 0, 0.33, 1]
        ),  # Weight of load shifting revenues
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

    config = {
        "n_years": 10,
        "battery_capacity_cost": (50, 115, 200),  # Cost per kWh
        "battery_installation_cost": (
            500,
            25,
        ),  # Installation cost per site (fixed cost, variable cost per kWh)
        "vpp_controller_cost": (500, 1500, 2500),  # Cost for VPP controller per
        "ffr_yield": 11840,  # Yield from FFR up and down market €/MW/year
        "fcr_yield": 119890,  # Yield from FCR-D up and down market €/MW/year
        "afrr_yield": 131354,  # Yield from aFRR up and down market €/MW/year
        "load_shifting_savings": 67882,  # Savings from load shifting €/MW/year
        "peak_shaving_savings_per_site": (1.35 * 2)
        * 12,  # Single site savings from peak shaving €/MW/year
        "connectivity_cost": (0, 12, 120),  # VPP connectivity cost per year
        "o&m_cost": (0.01, 0.02, 0.03),  # O&M cost as a fraction of investment cost
        "bsp_fee_dist": (
            0.1,
            0.2,
            0.3,
        ),  # BSP fee distribution parameters (min, mode, max)
        "site_mean_power": 2.5,  # Mean power per site in kW
        "vpp_total_power": 1000,  # Total power of the VPP in kW
        "discount_rate": (0.04, 0.057, 0.08),  # Discount rate for NPV calculation
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

    simulator = NPVSimulator(cases, config)
    sens_params, sens_array = simulator.run_sensitivity_analysis()
    scenarios = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c"]
    for i in range(len(scenarios)):
        plot_tornado(sens_params, sens_array, i, title=f"Scenario {scenarios[i]}")
    plot_tornado_grid(sens_params, sens_array, scenarios)

    results_with_different_site_power = []
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    # Boxplot for different site mean power scenarios
    config["site_mean_power"] = 5
    simulator = NPVSimulator(cases, config)
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)
    for i in range(len(scenarios)):
        npv_last_year = results[:, -1, i]
        print(f"Scenario {scenarios[i]} with site mean power 5 kW:")
        print(f"Median NPV: {np.median(npv_last_year)}\n")

    scenario = 4
    npv_last_year = results[:, -1, scenario]
    print(f"Median NPV: {np.median(npv_last_year)}")
    # print(f"Standard Deviation of NPV: {np.std(npv_last_year)}")
    print(f"Minimum NPV: {np.min(npv_last_year)}")
    print(f"Maximum NPV: {np.max(npv_last_year)}")
    print(f"5th Percentile NPV: {np.percentile(npv_last_year, 5)}")
    print(f"95th Percentile NPV: {np.percentile(npv_last_year, 95)}")
    print(cf[:, scenario])

    plt.figure(figsize=(10, 6))
    plt.hist(npv_last_year, bins=500, density=True)
    plt.xlabel("NPV (€)")
    plt.ylabel("Density")
    plt.show()

    plot_cumulative_npv(results, scenario, config["n_years"])

    config["site_mean_power"] = 7.5
    simulator = NPVSimulator(cases, config)
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    config["site_mean_power"] = 15
    simulator = NPVSimulator(cases, config)
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    plot_npv_boxplots(
        site_powers=[2.5, 5, 7.5, 15],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )

    plot_npv_hist_grid(
        site_powers=[2.5, 5, 7.5, 15],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )

    plot_npv_hist_overlaid(
        site_powers=[2.5, 5, 7.5, 15],
        results_with_different_site_power=results_with_different_site_power,
        scenarios=scenarios,
    )


def analysis():
    cases = {
        "Investment_size": np.array(
            [0, 1, 3, 0, 1, 3, 0, 1, 3]
        ),  # Extra capacity hours
        "FFR weight": np.array([0, 0, 0, 1, 0, 0, 1, 0, 0]),  # Weight of FFR revenues
        "FCR weight": np.array([0, 0, 0, 0, 1, 0, 0, 1, 0]),  # Weight of FCR revenues
        "aFRR weight": np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]),  # Weight of aFRR revenues
        "LS weight": np.array(
            [0, 0.33, 1, 0, 0.33, 1, 0, 0.33, 1]
        ),  # Weight of load shifting revenues
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

    config = {
        "n_years": 10,
        "battery_capacity_cost": (50, 115, 200),  # Cost per kWh
        "battery_installation_cost": (
            500,
            25,
        ),  # Installation cost per site (fixed cost, variable cost per kWh)
        "vpp_controller_cost": (500, 1500, 2500),  # Cost for VPP controller per
        "ffr_yield": 11840,  # Yield from FFR up and down market €/MW/year
        "fcr_yield": 119890,  # Yield from FCR-D up and down market €/MW/year
        "afrr_yield": 131354,  # Yield from aFRR up and down market €/MW/year
        "load_shifting_savings": 67882,  # Savings from load shifting €/MW/year
        "peak_shaving_savings_per_site": 24000
        / 200,  # (1.35 *  2) * 12,  # Single site savings from peak shaving €/MW/year
        "connectivity_cost": (0, 12, 120),  # VPP connectivity cost per year
        "o&m_cost": (0.01, 0.02, 0.03),  # O&M cost as a fraction of investment cost
        "bsp_fee_dist": (
            0.1,
            0.2,
            0.3,
        ),  # BSP fee distribution parameters (min, mode, max)
        "site_mean_power": 2.5,  # Mean power per site in kW
        "vpp_total_power": 1000,  # Total power of the VPP in kW
        "discount_rate": (0.04, 0.057, 0.08),  # Discount rate for NPV calculation
        "reserve_price_multiplier_dist": (
            -0.5,
            -0.2,
            0.5,
        ),  # Reserve price multiplier distribution (min, mode, max)
        "spot_volatility_multiplier_dist": (
            -0.0001,
            0.0,
            0.0001,
        ),  # (-0.1, 0, 0.1),  # Spot price multiplier distribution (min, mode, max)
        "power_charge_multiplier_dist": (
            -0.0001,
            0,
            0.0001,
        ),  # (-0.1, 0.2, 0.96),  # Power charge multiplier distribution (min, mode, max)
    }

    simulator = NPVSimulator(cases, config)
    sens_params, sens_array = simulator.run_sensitivity_analysis()
    scenarios = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c"]
    for i in range(len(scenarios)):
        plot_tornado(sens_params, sens_array, i, title=f"Scenario {scenarios[i]}")
    plot_tornado_grid(sens_params, sens_array, scenarios)

    results_with_different_site_power = []
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)

    # Boxplot for different site mean power scenarios
    config["site_mean_power"] = 5
    simulator = NPVSimulator(cases, config)
    results, cf, revs = simulator.run_uncertainty_analysis(count=3000000)
    results_with_different_site_power.append(results)
    for i in range(len(scenarios)):
        npv_last_year = results[:, -1, i]
        print(f"Scenario {scenarios[i]} with site mean power 5 kW:")
        print(f"Median NPV: {np.median(npv_last_year)}\n")

    scenario = 4
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
    for scenario in range(len(scenarios)):
        print(f"Revenues {scenarios[scenario]}:")
        print(revs[:, scenario])
        print(f"Cash Flows: {cf[:, scenario]}")
        print("\n")


if __name__ == "__main__":
    main()

# %%
