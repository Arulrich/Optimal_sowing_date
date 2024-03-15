# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:33:18 2024

@author: arulrich
"""

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray
import seaborn as sns



# open it
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')

region = "Zambia"
# load simulated yield data as dataset!!

model = 'ensemble'

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'
ds_opt = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
ds_cal = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'
ds_opt = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
ds_cal = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'
ds_opt = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
ds_cal = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
ds_opt = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
ds_cal = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')


################ clip dataset according to shapefiles ###################

def clip_dset(dataset, shape):
    """Clip a dataset with the information from a shapefile

    Args:
        dataset (xr.dataset): dataset to be clipped
        shape (geopandas shapefile): gpd object with polygon or multipolygon defining the area to be clipped

    Returns:
        clip_dset: a dataset where there are data only within the shape's area, elsewhere is nan
    """
    dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    dataset.rio.write_crs("epsg:4326", inplace=True)
    clip_dset = dataset.rio.clip(shape.geometry.apply(mapping), shape.crs, drop=False)
    return clip_dset


""" already did it

# create a df with yields and their categories source/AEZ

df = pd.DataFrame(columns=['yield', 'Source', 'AEZ'])

source = ['Crop_Calendar', 'Opt_sim']

AEZ = aez['AEZ'].unique()

for x in source:
    if x == 'Crop_Calendar':
        ds = ds_cal
        df_temp = pd.DataFrame(columns=['yield', 'Source', 'AEZ'])
        for y in AEZ:
            print(x, "and", y)
            df_temp_AEZ = pd.DataFrame(columns=['yield', 'Source', 'AEZ'])
            subset = aez[aez['AEZ'] == y]
            da = clip_dset(ds, subset)['Yield_average']
            df_temp_AEZ['yield'] = da.to_dataframe(name='yield').reset_index()['yield']
            df_temp_AEZ['AEZ'] = y
            df_temp_AEZ['Source'] = x
            df_temp = pd.concat([df_temp, df_temp_AEZ], axis=0, ignore_index=True)
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
    elif x == 'Opt_sim':
        ds = ds_opt
        df_temp = pd.DataFrame(columns=['yield', 'Source', 'AEZ'])
        for y in AEZ:
            print(x, "and", y)
            df_temp_AEZ = pd.DataFrame(columns=['yield', 'Source', 'AEZ'])
            subset = aez[aez['AEZ'] == y]
            da = clip_dset(ds, subset)['Yield_average']
            df_temp_AEZ['yield'] = da.to_dataframe(name='yield').reset_index()['yield']
            df_temp_AEZ['AEZ'] = y
            df_temp_AEZ['Source'] = x
            df_temp = pd.concat([df_temp, df_temp_AEZ], axis=0, ignore_index=True)
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
     
    
df.dropna(inplace=True)

df['group'] = df['Source'] + '_' + df['AEZ']  

df = df.loc[df['AEZ'] != 'No_data']

df['AEZ'] = df['AEZ'].apply(lambda x: '\n'.join(x.split('_')))

df.to_csv(save_dir+ '/Zam_yield_per_source_AEZ.csv')
"""

# load it
df = pd.read_csv(save_dir+ '/Zam_yield_per_source_AEZ.csv')


sns.boxplot(x="AEZ", y="yield", hue="Source", data=df, legend='auto')
plt.title(region, {'fontsize': 15})
plt.xlabel(' ', {'fontsize': 12})
plt.ylabel('Yield [t/ha]', {'fontsize': 12})  
plt.ylim((0,13))
plt.rc('xtick', labelsize=8)

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/yield_AEZ_source_WA.png', dpi=260)

plt.show()


