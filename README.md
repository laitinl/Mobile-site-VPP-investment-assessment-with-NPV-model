# DC-Waste-Heat-investment-assessment-with-NPV-model
A tool for assessing various sized DCs investment in waste heat recapturing equipment 

## 1. Overview 

This tool calculates NPV value for waste heat utilization in different sized data centers. 

### 1.1. About NPV simulator

We investigated the simultaneous effect of variability of the most significant factors with a Monte Carlo simulation on the five most significant factors affecting the NPV model outcome. The Monte Carlo simulation generates multiple scenarios for representing possible realizations of uncertainties. We modeled a simplified NPV model for the simulation. The factors included in the simulation were heat price, electricity price, COP, the rate of return and total investment. We made the following simplifications to the NPV model: 1) tax shield is not used, 2) all values are averaged and aggregated to year level, 3) the four separate parts of the investment project are aggregated to one total investment, 4) ramp-up time is taken into account as an average number of racks during a 10 year period, and 5) the medium case was used to populate other case-specific factors with multipliers to two remaining cases.

## 3. Parameters

### 3.1. Parameter Taxonomy

Parameter | Example Value | Uncertainty | Description
-------|---------|---------|---------
no_of_years | 10 | - | Number of years for the NPV simulation
heat_recovery_rate | 0.97 | - | The amount of heat captured in percentages
to_gwh | 720/1000000 | - | Conversion to GWh
salary_employer_cost_factor | 0.144 | - | The factor by which salary is multiplied to get employer costs
maintenance_salary | 26058 | - | Maintenence worker salary in a year
business_manager_salary | 38934 | - | Business manager salary in a year
tax_rate | 0.2234 | - | Corporate tax
montly_growth_rate | 0.0725 | - | The monthly growth rate
investment_total_base | 775000 | - | The investment size in the base case medium
small_investment_mul | 0.148 | - | The investment multiplied for small case
med_investment_mul | 1 | - | The investment multiplied for medium case
large_investment_mul | 9.510 | - | The investment multiplied for large case
small_electricity_mul | 1.025 | - | The electricity price multiplier for small case
medium_electricity_mul | 1 | - | The electricity price multiplier for medium case
large_electricity_mul | 0.773 | - | The electricity price multiplier for large case
marketing_small_mul | 0.05 | - | Marketing cost multiplier for small case
marketing_med_mul | 0.05 | - | Marketing cost multiplier for medium case
marketing_large_mul | 0.05 | - | Marketing cost multiplier for large case
maintenance_base | 9600 | - | Yearly maintenance cost
maintenance_small_mul | 0.25 | - | Yearly maintenance cost multiplier for small case
maintenance_med_mul | 1 | - | Yearly maintenance cost multiplier for medium case
maintenance_large_mul | 10 | - | Yearly maintenance cost multiplier for large case
rent_base | 18000 | - | The cost of rented premises
rent_small_mul | 0.3333 | - | Yearly rent cost multiplier for small case
rent_med_mul | 1 | - | Yearly rent cost multiplier for medium case
rent_large_mul | 10 | - | Yearly rent cost multiplier for large case
rate_of_return | 0.12-0.18 range | - |  Rete of return for investment
COP | 2.8125-4.6875 range | - |  COP range
investment_cost | 0.8-1.2 range | - |  Investment range
electricity_price_base | 64510-96766 range | - |  Electricity price range
heat_price_base | 36720-55080 range | - |  Heat price range
