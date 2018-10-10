
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from astetik import dist

def npv_simulation(count):
    
    out = []
    
    for i in range(count):
        if i <= count:

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

            small_staff_mul = .4
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
            investment_cost = np.arange(0.8,1.2,0.01)
            electricity_price_base = np.arange(64510,96766,10)
            heat_price_base = np.arange(36720, 55080, 10)

            capital_investment = investment_total * investment_total_base * random.choice(investment_cost)
            dep_amortization = capital_investment / 10

            total_power_consumed = initial_racks * power_per_rack_kwh * to_gwh
            total_for_priming = total_power_consumed / random.choice(COP) * heat_recovery_rate
            heat_captured = total_power_consumed + total_for_priming * heat_recovery_rate
            heat_price = random.choice(heat_price_base)
            total_revenue = heat_captured * heat_price * 12

            maintenance_people = maintenance_salary * staff_price * (1 + salary_employer_cost_factor)
            business_people = business_manager_salary * staff_price * (1 + salary_employer_cost_factor)
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

            round_out = np.npv(random.choice(rate_of_return), OFCF)
            round_out = int(round_out)
                       
            out.append([round_out, initial_racks])
    
    out = pd.DataFrame(out)
    out.columns = ['NPV','racks']
    return pd.DataFrame(out)


# In[2]:


scores = npv_simulation(3000000)


# In[3]:


dist('NPV', scores[scores.racks == 45], bins=500, color='blue', title='NPV Distribution Small Cases')


# In[4]:


scores[scores.racks == 45]['NPV'].mean() 


# In[5]:


scores[scores.racks == 45]['NPV'].std() 


# In[6]:


scores[scores.racks == 45]['NPV'].min() 


# In[7]:


scores[scores.racks == 45]['NPV'].max() 


# In[8]:


dist('NPV', scores[scores.racks == 450], bins=500, title='NPV Distribution Medium Cases')


# In[9]:


scores[scores.racks == 450]['NPV'].mean() 


# In[10]:


scores[scores.racks == 450]['NPV'].std() 


# In[11]:


scores[scores.racks == 450]['NPV'].min() 


# In[12]:


scores[scores.racks == 450]['NPV'].max() 


# In[13]:


dist('NPV', scores[scores.racks == 4500], bins=300, color='green', title='NPV Distribution Large Cases')


# In[14]:


scores[scores.racks == 4500]['NPV'].mean() 


# In[15]:


scores[scores.racks == 4500]['NPV'].std() 


# In[16]:


scores[scores.racks == 4500]['NPV'].min() 


# In[17]:


scores[scores.racks == 4500]['NPV'].max() 

