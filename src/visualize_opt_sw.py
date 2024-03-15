# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:10:03 2023

@author: arulrich
"""

# extraction of mean and yield cv along sowing dates for single locations to get an idea of the time of the growing season
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd


def df_location_max_yield(da, lat, lon):
    """Extract for a single location the mean yield variation along the 36 sowing dates.

    Args:
        da (dataarray): contains averaged yield values for each pixel (lat, lon, sowing_date)
        lat, lon (float): values for longitude and latitude
        
    Returns:
        df (dataframe): dataframe containing the average yield for each sowing date
    """
    vp = da.sel(lat = lat, lon = lon, method='nearest')
    
    # Create a DataFrame
    df = vp.to_dataframe(name='mean_yield').reset_index()
    
    # Add 'lat' and 'lon' columns
    df['lat'] = vp.coords['lat'].values
    df['lon'] = vp.coords['lon'].values
    
    # Rename 'sowing_date' to the appropriate column name
    df = df.rename(columns={'sowing_date': 'sowing_date'})
    return df

def df_location_min_cv(da, lat, lon):
    """Extract for a single location the cv yield variation along the 36 sowing dates.

    Args:
        da (dataarray): contains cv values for each pixel (lat, lon, sowing_date)
        lat, lon (float): values for longitude and latitude
        
    Returns:
        df (dataframe): dataframe containing the cv for each sowing date
    """
    vp = da.sel(lat = lat, lon = lon, method='nearest')
    
    # Create a DataFrame
    df = vp.to_dataframe(name='CV').reset_index()
    
    # Add 'lat' and 'lon' columns
    df['lat'] = vp.coords['lat'].values
    df['lon'] = vp.coords['lon'].values
    
    # Rename 'sowing_date' to the appropriate column name
    df = df.rename(columns={'sowing_date': 'sowing_date'})
    return df

def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x

# chose directory
# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'



plt.figure()
des_lat = -21
des_lon = 30
for i in range(0,4):
    model =['ensemble', 'dssat', 'stics', 'celsius'][i]
    da_y = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
    max_yield = df_location_max_yield(da = da_y, lat = des_lat, lon = des_lon)
    max_yield = max_yield.fillna(0)
    max_yield.sort_values(by='sowing_date', inplace=True)
    plt.plot(max_yield['sowing_date'], max_yield['mean_yield'], label = model, color = ['grey','red', 'green', 'blue'][i])
    """
    da_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
    min_cv = df_location_min_cv(da_cv, des_lat, des_lon)
    plt.plot(min_cv['sowing_date'], min_cv['CV'], label = model, color = ['grey','red', 'green', 'blue'][i], linestyle='dashed')
    """
    plt.title('Mean yield: ' + str(des_lat) + '°N, ' + str(des_lon) +'°E')
    plt.xlabel('Sowing dates [DOY]')
    plt.ylabel('Yield in t')
    plt.legend()
plt.show()


# look at the plots and define the start of the season for a better visualization (it shifts the plots if we have a season between two years)
start_season = 150 # no shift with start_of_the_season = 0



plt.figure()
for i in range(0,4):
    model =['ensemble', 'dssat', 'stics', 'celsius'][i]
    da_y = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
    max_yield = df_location_max_yield(da = da_y, lat = des_lat, lon = des_lon)
    max_yield = max_yield.fillna(0)
    max_yield['sowing_date'] = max_yield['sowing_date'].apply(adjust_start)
    max_yield = max_yield.sort_values(by='sowing_date')
    plt.plot(max_yield['sowing_date'], max_yield['mean_yield'], label = 'Yield mean', color = ['black','red', 'green', 'blue'][i])
    
    da_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
    min_cv = df_location_min_cv(da_cv, des_lat, des_lon)
    min_cv['sowing_date'] = min_cv['sowing_date'].apply(adjust_start)
    min_cv = min_cv.sort_values(by = 'sowing_date')
    plt.plot(min_cv['sowing_date'], min_cv['CV'], label = 'Yield CV', color = ['black','red', 'green', 'blue'][i], linestyle='dashed')
    
    plt.title('Location: ' + str(des_lat) + '°N, ' + str(des_lon) +'°E', {'fontsize': 15})
    plt.xlabel('Sowing dates [doy]', {'fontsize': 12})
    plt.ylabel('Yield [t]', {'fontsize': 12}) 
    plt.legend()
    
    plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/opt_sw_definition.png', dpi=260)

plt.show()

# look at the models shape with the relative yields values to their maximum

plt.figure()
for i in range(0,4):
    model =['ensemble', 'dssat', 'stics', 'celsius'][i]
    da_y = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
    max_yield = df_location_max_yield(da = da_y, lat = des_lat, lon = des_lon)
    max_yield = max_yield.fillna(0)
    max_yield['sowing_date'] = max_yield['sowing_date'].apply(adjust_start)
    max_yield = max_yield.sort_values(by='sowing_date')
    max_yield['mean_yield_rel'] = max_yield['mean_yield']/max_yield['mean_yield'].max()
    plt.plot(max_yield['sowing_date'], max_yield['mean_yield_rel'], label = model, color = ['grey','red', 'green', 'blue'][i])
    """
    da_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
    min_cv = df_location_min_cv(da_cv, des_lat, des_lon)
    min_cv['sowing_date'] = min_cv['sowing_date'].apply(adjust_start)
    min_cv = min_cv.sort_values(by = 'sowing_date')
    plt.plot(min_cv['sowing_date'], min_cv['CV'], label = model, color = ['grey','red', 'green', 'blue'][i], linestyle='dashed')
    """
    plt.title('Relative mean yield: ' + str(des_lat) + '°N, ' + str(des_lon) +'°E')
    plt.xlabel('Sowing dates [DOY]')
    plt.ylabel('Relative Yield')
    plt.legend()
plt.show()



# if it looks good then adjust the start for the entire dataarray and visualize them with shp
# plot optimal sowing date with country and county borders

# load political boundaries
# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'

region = "Zambia"

if region == "Zimbabwe":
    
    # Zimbabwe
    gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp') # Zimbabwe
    gdf02 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_0.shp') # Mozambique
    gdf03 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp') # Zambia
    gdf04 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_0.shp') # South Afrika
    gdf05 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_0.shp') # Botswana
    
    gdf_all_nat = pd.concat([gdf01, gdf02, gdf03, gdf04, gdf05], ignore_index=True)
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15], ignore_index=True)
    
elif region == 'Zambia': 
    
    # Zambia
    gdf01 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp') # Zambia
    gdf02 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp') # Zimbabwe
    gdf03 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_0.shp') # Mozambique
    gdf04 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_0.shp') # Botswana
    gdf05 = gpd.read_file(dir_shp + 'gadm41_MWI_shp/gadm41_MWI_0.shp') # Malawi
    gdf06 = gpd.read_file(dir_shp + 'gadm41_NAM_shp/gadm41_NAM_0.shp') # Namibia
    gdf07 = gpd.read_file(dir_shp + 'gadm41_AGO_shp/gadm41_AGO_0.shp') # Angola
    gdf08 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_0.shp') # DR Congo
    gdf09 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_0.shp') # Tanzania
    
    gdf_all_nat = pd.concat([gdf01, gdf02, gdf03, gdf04, gdf05, gdf06, gdf07, gdf08, gdf09], ignore_index=True)
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_MWI_shp/gadm41_MWI_1.shp') 
    gdf16 = gpd.read_file(dir_shp + 'gadm41_NAM_shp/gadm41_NAM_1.shp')
    gdf17 = gpd.read_file(dir_shp + 'gadm41_AGO_shp/gadm41_AGO_1.shp')
    gdf18 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_1.shp')
    gdf19 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15, gdf16, gdf17, gdf18, gdf19], ignore_index=True)

elif region == "West Africa":

    # West Africa
    gdf01 = gpd.read_file(dir_shp + 'gadm41_BEN_shp/gadm41_BEN_0.shp') # Benin
    gdf02 = gpd.read_file(dir_shp + 'gadm41_BFA_shp/gadm41_BFA_0.shp') # Burkina Faso
    gdf03 = gpd.read_file(dir_shp + 'gadm41_CIV_shp/gadm41_CIV_0.shp') # Côte d'Ivoire
    gdf04 = gpd.read_file(dir_shp + 'gadm41_GHA_shp/gadm41_GHA_0.shp') # Ghana
    gdf05 = gpd.read_file(dir_shp + 'gadm41_GIN_shp/gadm41_GIN_0.shp') # Guinea
    gdf06 = gpd.read_file(dir_shp + 'gadm41_LBR_shp/gadm41_LBR_0.shp') # Liberia
    gdf07 = gpd.read_file(dir_shp + 'gadm41_MLI_shp/gadm41_MLI_0.shp') # Mali
    gdf08 = gpd.read_file(dir_shp + 'gadm41_NER_shp/gadm41_NER_0.shp') # Niger
    gdf09 = gpd.read_file(dir_shp + 'gadm41_SEN_shp/gadm41_SEN_0.shp') # Senegal
    gdf010 = gpd.read_file(dir_shp + 'gadm41_TGO_shp/gadm41_TGO_0.shp') # Togo
    
    gdf_all_nat = pd.concat([gdf01, gdf02, gdf03, gdf04, gdf05, gdf06, gdf07, gdf08, gdf09, gdf010], ignore_index=True)
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_BEN_shp/gadm41_BEN_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_BFA_shp/gadm41_BFA_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_CIV_shp/gadm41_CIV_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_GHA_shp/gadm41_GHA_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_GIN_shp/gadm41_GIN_1.shp')
    gdf16 = gpd.read_file(dir_shp + 'gadm41_LBR_shp/gadm41_LBR_1.shp')
    gdf17 = gpd.read_file(dir_shp + 'gadm41_MLI_shp/gadm41_MLI_1.shp')
    gdf18 = gpd.read_file(dir_shp + 'gadm41_NER_shp/gadm41_NER_1.shp')
    gdf19 = gpd.read_file(dir_shp + 'gadm41_SEN_shp/gadm41_SEN_1.shp')
    gdf110 = gpd.read_file(dir_shp + 'gadm41_TGO_shp/gadm41_TGO_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15, gdf16, gdf17, gdf18, gdf19, gdf110], ignore_index=True)

elif region == "East Africa":
    
    # East Africa
    gdf01 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_0.shp') # DR Congo
    gdf02 = gpd.read_file(dir_shp + 'gadm41_BDI_shp/gadm41_BDI_0.shp') # Burundi
    gdf03 = gpd.read_file(dir_shp + 'gadm41_KEN_shp/gadm41_KEN_0.shp') # Kenia
    gdf04 = gpd.read_file(dir_shp + 'gadm41_RWA_shp/gadm41_RWA_0.shp') # Rwanda
    gdf05 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_0.shp') # Tanzania
    gdf06 = gpd.read_file(dir_shp + 'gadm41_UGA_shp/gadm41_UGA_0.shp') # Uganda
    
    gdf_all_nat = pd.concat([gdf01, gdf02, gdf03, gdf04, gdf05, gdf06], ignore_index=True)
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_BDI_shp/gadm41_BDI_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_KEN_shp/gadm41_KEN_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_RWA_shp/gadm41_RWA_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_1.shp')
    gdf16 = gpd.read_file(dir_shp + 'gadm41_UGA_shp/gadm41_UGA_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15, gdf16], ignore_index=True)


# load agroecological zones
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')


# load opt_sw data
model = 'ensemble'
da = xr.open_dataarray(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')


da.min()
da.max()
da.quantile(0.05)
da.quantile(0.95)

df = da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=36)

# adjust start of the season for the entire area
start_season = 170 #specific for each region: 250 for Zim, 170 for Zam, 0 for EA and WA

da = xr.apply_ufunc(adjust_start,da)

da.min()
da.max()
da.quantile(0.05)
da.quantile(0.95)

df = da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=24)
plt.ylim((0,50))


fig, ax = plt.subplots(figsize=(10, 8))

da.plot(cmap='plasma', vmin =260, vmax=410)

plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='gray')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/opt_sw_Zam.png', dpi=260)

plt.show()


