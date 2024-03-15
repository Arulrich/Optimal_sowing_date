# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:18:01 2023

@author: arulrich
"""
# run this in another environment where you have xarray v >= 0.19.0
# I run it directly in the anaconda prompt in the environment spatial2

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

"""
# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
"""
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

model = 'dssat'

opt_sw = pd.read_csv(save_dir + f'/{model}_MgtMais0_optSwDateMain_2.0.csv')
opt_sw = opt_sw[['lat', 'lon', 'sowing_date']]
opt_sw = opt_sw.set_index(['lat', 'lon']) # set lat and lon as indeces
result_da = opt_sw.to_xarray() # works with xarray v 0.19.0
print(result_da)
result_da['sowing_date'].plot()
plt.show()
result_da.to_netcdf(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

