# %%


import numpy as np
import numpy_financial as npf
import pandas as pd
import random
from astetik import hist
import matplotlib.pyplot as plt


class NPVSimulator:
    """
    A class to simulate Net Present Value (NPV) for different mobile site
    virtual power plant (VPP) battery investment scenarios.
    """

    def __init__(self, config: dict):
        """
        Initialize the NPV simulator with a configuration dictionary.

        Args:
        config (dict): Configuration parameters for the simulation.
        """
        self.config = config

    def run(self, count: int, random_seed: int = 42):
        """
        Run the NPV simulation.

        Args:
        count (int): Number of simulations to run.
        random_state (int): Seed for random number generation.
        """
        np.random.seed(random_seed)
        out = np.zeros((count, 1))

        # Fixed parameters
        n_years = self.config["n_years"]
        battery_capacity_cost = self.config["battery_capacity_cost"]
        battery_installation_cost = self.config["battery_installation_cost"]
        vpp_controller_cost = self.config["vpp_controller_cost"]
        reserve_market_yield = self.config["reserve_market_yield"]
        load_shifting_savings = self.config["load_shifting_savings"]
        site_mean_power = self.config["site_mean_power"]
        vpp_total_power = self.config["vpp_total_power"]
        discount_rate = self.config["discount_rate"]

        # Derived parameters
        n_sites = vpp_total_power // site_mean_power
        battery_capacity = vpp_total_power  # Assuming battery capacity equals VPP total power for one hour
        investment_cost = (
            battery_capacity * battery_capacity_cost
            + n_sites * battery_installation_cost
            + n_sites * vpp_controller_cost
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
        cash_flows = np.zeros((count, n_years + 1))
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
            cash_flows[:, year] = total_revenue * (1 - bsp_fee)

        # Calculate NPV
        for i in range(count):
            out[i] = npf.npv(discount_rate, cash_flows[i])

        return out


# %%
def main():
    config = {
        "n_years": 10,
        "battery_capacity_cost": 200,  # Cost per kWh
        "battery_installation_cost": 1000,  # Installation cost per site
        "vpp_controller_cost": 1000,  # Cost for VPP controller per
        "reserve_market_yield": 108000,  # Yield from reserve market €/MW/year
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
    }

    simulator = NPVSimulator(config)
    results = simulator.run(count=3000000)
    print(f"Mean NPV: {np.mean(results)}")
    print(f"Standard Deviation of NPV: {np.std(results)}")
    print(f"Minimum NPV: {np.min(results)}")
    print(f"Maximum NPV: {np.max(results)}")
    print(f"5th Percentile NPV: {np.percentile(results, 5)}")
    print(f"95th Percentile NPV: {np.percentile(results, 95)}")

    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=500, density=True)
    plt.xlabel("NPV (€)")
    plt.ylabel("Density")
    plt.show()


if __name__ == "__main__":
    main()

# %%
"""
def npv_simulation(count):

    out = []

    for i in range(count):
        # model years
        no_of_years = 10

        # core parameters
        heat_recovery_rate = 0.97
        power_per_rack_kwh = 4.286
        to_gwh = 720 / 1000000

        # staff related parameters
        salary_employer_cost_factor = 0.144
        maintenance_salary = 26058
        business_manager_salary = 38934

        small_staff_mul = 0.4
        med_staff_mul = 1
        large_staff_mul = 2

        # other business parameters
        depreciation_years = 10
        tax_rate = 0.2234
        monthly_growth_rate = 0.0725

        # investment related
        investment_total_base = 775000
        small_investment_mul = 0.148
        med_investment_mul = 1
        large_investment_mul = 9.510

        # electricity price related
        small_electricity_mul = 1.025
        med_electricity_mul = 1
        large_electricity_mul = 0.773

        # marketing
        marketing_small_mul = 0.05
        marketing_med_mul = 0.02
        marketing_large_mul = 0.005

        # maintenance
        maintenance_base = 9600
        maintenance_small_mul = 0.25
        maintenance_med_mul = 1
        maintenance_large_mul = 10

        # rent and others
        rent_base = 18000
        rent_small_mul = 0.3333
        rent_med_mul = 1
        rent_large_mul = 10

        # case rack sizes
        cases = [45, 450, 4500]

        # first picking the size
        initial_racks = random.choice(cases)

        # setting the size specific parameters

        if initial_racks == 45:
            electicity_price = small_electricity_mul
            staff_price = small_staff_mul
            investment_total = small_investment_mul

        elif initial_racks == 450:
            electicity_price = med_electricity_mul
            staff_price = med_staff_mul
            investment_total = med_investment_mul

        elif initial_racks == 4500:
            electicity_price = large_electricity_mul
            staff_price = large_staff_mul
            investment_total = large_investment_mul

        # then picking the rest of parameters
        rate_of_return = np.arange(0.12, 0.18, 0.001)
        COP = np.arange(2.8125, 4.6875, 0.1)
        investment_cost = np.arange(0.8, 1.2, 0.01)
        electricity_price_base = np.arange(64510, 96766, 10)
        heat_price_base = np.arange(36720, 55080, 10)

        capital_investment = (
            investment_total * investment_total_base * random.choice(investment_cost)
        )
        dep_amortization = capital_investment / 10

        total_power_consumed = initial_racks * power_per_rack_kwh * to_gwh
        total_for_priming = (
            total_power_consumed / random.choice(COP) * heat_recovery_rate
        )
        heat_captured = total_power_consumed + total_for_priming * heat_recovery_rate
        heat_price = random.choice(heat_price_base)
        total_revenue = heat_captured * heat_price * 12

        maintenance_people = (
            maintenance_salary * staff_price * (1 + salary_employer_cost_factor)
        )
        business_people = (
            business_manager_salary * staff_price * (1 + salary_employer_cost_factor)
        )
        electricity_price = electicity_price * random.choice(electricity_price_base)
        cost_of_priming = total_for_priming * electricity_price * 12
        total_cogs = maintenance_people + business_people + cost_of_priming

        # setting the size specific parameters

        if initial_racks == 45:
            marketing_cost = marketing_small_mul * total_revenue
            maintenance_cost = maintenance_base * maintenance_small_mul
            rent_cost = rent_base * rent_small_mul

        elif initial_racks == 450:
            marketing_cost = marketing_med_mul * total_revenue
            maintenance_cost = maintenance_base * maintenance_med_mul
            rent_cost = rent_base * rent_med_mul

        elif initial_racks == 4500:
            marketing_cost = marketing_large_mul * total_revenue
            maintenance_cost = maintenance_base * maintenance_large_mul
            rent_cost = rent_base * rent_large_mul

        OFCF = []

        # investment year
        OFCF.append(0 - (investment_total_base * investment_total))

        # second year onwards

        for i in range(no_of_years):
            investment = 0
            gross_profit = total_revenue - total_cogs
            other_costs = marketing_cost + maintenance_cost + rent_cost
            EBITDA = gross_profit - other_costs
            EBIT = EBITDA - dep_amortization

            if EBIT <= 0:
                taxes = 0
            elif EBIT > 0:
                taxes = EBIT * tax_rate

            NOPAT = EBIT - taxes
            OFCF.append(NOPAT + dep_amortization)

        OFCF = pd.Series(OFCF).astype(int)

        round_out = npf.npv(random.choice(rate_of_return), OFCF)
        round_out = int(round_out)

        out.append([round_out, initial_racks])

    out = pd.DataFrame(out)
    out.columns = ["NPV", "racks"]
    return pd.DataFrame(out)


# In[2]:


scores = npv_simulation(30000)


# In[3]:


hist(
    scores[scores.racks == 45],
    "NPV",
    bins=500,
    style="astetik",
    # color="blue",
    # title="NPV Distribution Small Cases",
)


# In[4]:


scores[scores.racks == 45]["NPV"].mean()


# In[5]:


scores[scores.racks == 45]["NPV"].std()


# In[6]:


scores[scores.racks == 45]["NPV"].min()


# In[7]:


scores[scores.racks == 45]["NPV"].max()


# In[8]:


hist(
    scores[scores.racks == 450],
    "NPV",
    bins=500,  # title="NPV Distribution Medium Cases"
)


# In[9]:


scores[scores.racks == 450]["NPV"].mean()


# In[10]:


scores[scores.racks == 450]["NPV"].std()


# In[11]:


scores[scores.racks == 450]["NPV"].min()


# In[12]:


scores[scores.racks == 450]["NPV"].max()


# In[13]:


hist(
    scores[scores.racks == 4500],
    "NPV",
    bins=500,
    # color="green",
    # title="NPV Distribution Large Cases",
)


# In[14]:


scores[scores.racks == 4500]["NPV"].mean()


# In[15]:


scores[scores.racks == 4500]["NPV"].std()


# In[16]:


scores[scores.racks == 4500]["NPV"].min()


# In[17]:


scores[scores.racks == 4500]["NPV"].max()
"""
