# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:57:42 2024

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

region = "West Africa"
# load opt_sw data as dataset!!

model = 'ensemble'

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'
ds_Zim = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'
ds_WA = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'
ds_Zam = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
ds_EA = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

"""

# load the right calendar according to the region
calendar_ds = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zim_crop_calendar_0_05.nc') # Zimbabwe
calendar_ds = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/EA_crop_calendar_0_05.nc') # East Africa
calendar_ds = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/WA_crop_calendar_0_05.nc') # West Africa
calendar_ds = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zam_crop_calendar_0_05.nc') # Zambia


# adjust start of the season for the entire area
def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x

start_season = 0 #specific for each region: 250 for Zim, 170 for Zam, 0 for EA and WA
ds_WA = xr.apply_ufunc(adjust_start,ds_WA)
da = ds_WA['sowing_date']

calendar_ds = xr.apply_ufunc(adjust_start, calendar_ds)
calendar = calendar_ds['planting_day']
"""

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

"""
# create a df with sowing date and their categories source/AEZ

df = pd.DataFrame(columns=['sowing_date', 'source', 'AEZ'])

source = ['Crop_Calendar', 'Optimal_sowing_date']

AEZ = aez['AEZ'].unique()

for x in source:
    if x == 'Crop_Calendar':
        ds = calendar_ds
        df_temp = pd.DataFrame(columns=['sowing_date', 'source', 'AEZ'])
        for y in AEZ:
            print(x, "and", y)
            df_temp_AEZ = pd.DataFrame(columns=['sowing_date', 'source', 'AEZ'])
            subset = aez[aez['AEZ'] == y]
            da = clip_dset(ds, subset)['planting_day']
            df_temp_AEZ['sowing_date'] = da.to_dataframe(name='sowing_date').reset_index()['sowing_date']
            df_temp_AEZ['AEZ'] = y
            df_temp_AEZ['source'] = x
            df_temp = pd.concat([df_temp, df_temp_AEZ], axis=0, ignore_index=True)
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
    elif x == 'Optimal_sowing_date':
        ds = ds_WA
        df_temp = pd.DataFrame(columns=['sowing_date', 'source', 'AEZ'])
        for y in AEZ:
            print(x, "and", y)
            df_temp_AEZ = pd.DataFrame(columns=['sowing_date', 'source', 'AEZ'])
            subset = aez[aez['AEZ'] == y]
            da = clip_dset(ds, subset)['sowing_date']
            df_temp_AEZ['sowing_date'] = da.to_dataframe(name='sowing_date').reset_index()['sowing_date']
            df_temp_AEZ['AEZ'] = y
            df_temp_AEZ['source'] = x
            df_temp = pd.concat([df_temp, df_temp_AEZ], axis=0, ignore_index=True)
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
     
    
df.dropna(inplace=True)

df['group'] = df['source'] + '_' + df['AEZ']  

df = df.loc[df['AEZ'] != 'No_data']

df.to_csv(save_dir+ '/WA_sw_per_source_AEZ.csv')
"""

# load it
df = pd.read_csv(save_dir+ '/WA_sw_per_source_AEZ.csv')

df.loc[df['source']=='Optimal_sowing_date','source'] = 'Opt_sim'

# df= df.loc[df['AEZ'] != 'Tropic_cool_arid'] # just for East Africa

df['AEZ'] = df['AEZ'].apply(lambda x: '\n'.join(x.split('_')))

df.rename(columns={'source': 'Source'}, inplace = True)


fig, ax = plt.subplots(figsize=(10, 8))

sns.violinplot(x="AEZ", y="sowing_date", hue="Source", data=df, legend='auto')
plt.title(region, {'fontsize': 15})
plt.xlabel(' ', {'fontsize': 12})
plt.ylabel('Sowing date [doy]', {'fontsize': 12})  
#plt.ylim((280,420))
plt.rc('xtick', labelsize=9)


plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/sw_AEZ_source_WA.png', dpi=260)


plt.show()


