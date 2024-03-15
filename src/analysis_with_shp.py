# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:22:17 2024

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

def diff_sw_date(swa,swb):
    """Determines the difference between two sowing dates.

    Args:
        swa,swb (int): sowing dates in doy

    Returns:
        diff: difference between the two sowing dates in days
    """
    sowing = [swa,swb]
    #diff = np.min([np.max(sowing, axis=0) - np.min(sowing, axis=0), 365 + np.min(sowing, axis=0) - np.max(sowing, axis=0)], axis=0)

    diff = min([max(sowing)-min(sowing), 365+min(sowing)-max(sowing)])
    return diff
##################### load data #############################
""" already did it
# open SSA AEZ shapefile
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez.shp', crs="epsg:4326")
aez.plot()
aez

# Display basic information of gpd shapefile
print("Columns:", aez.columns)
print("Data types:", aez.dtypes)
print("Number of rows and columns:", aez.shape)
print("First few rows:")
print(aez.head())

# add column with the name of the AEZ
aez["AEZ"] = float('nan')


# with SSA AEZ
reg = aez['DN'].unique()

for i in range(len(reg)):
    plt.figure()
    subset = aez[aez['DN'] == reg[i]]
    subset.plot()
    #da = clip_dset(ds, subset)['sowing_date']
    #da.plot(vmin=0, vmax=365)
    plt.title(f'{reg[i]}')
    plt.xlim((-22,57))
    plt.ylim((-38,30))

aez['AEZ'][aez['DN'] == 0] = 'No_data'
# aez['AEZ'][aez['DN'] == 0] = 'Subtropic - warm / arid'
aez['AEZ'][aez['DN'] == 1] = 'Subtropic_warm_semiarid'
aez['AEZ'][aez['DN'] == 2] = 'Subtropic_warm_subhumid'
aez['AEZ'][aez['DN'] == 3] = 'Subtropic_warm_humid'
aez['AEZ'][aez['DN'] == 4] = 'Subtropic_cool_arid'
aez['AEZ'][aez['DN'] == 5] = 'Subtropic_cool_semiarid'
aez['AEZ'][aez['DN'] == 6] = 'Subtropic_cool_subhumid'
aez['AEZ'][aez['DN'] == 7] = 'Tropic_warm_arid'
aez['AEZ'][aez['DN'] == 8] = 'Tropic_warm_semiarid'
aez['AEZ'][aez['DN'] == 9] = 'Tropic_warm_subhumid'
aez['AEZ'][aez['DN'] == 10] = 'Tropic_warm_humid'
aez['AEZ'][aez['DN'] == 11] = 'Tropic_cool_arid'
aez['AEZ'][aez['DN'] == 12] = 'Tropic_cool_semiarid'
aez['AEZ'][aez['DN'] == 13] = 'Tropic_cool_subhumid'
aez['AEZ'][aez['DN'] == 14] = 'Tropic_cool_humid'

aez.to_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')
"""
# open it
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')

"""
# special shp for Zim
# open SSA AEZ shapefile
aezZim = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_Zim\\AEZ_Zimbabwe.shp', crs="epsg:4326")
aezZim.plot()
aezZim

# Display basic information of gpd shapefile
print("Columns:", aezZim.columns)
print("Data types:", aezZim.dtypes)
print("Number of rows and columns:", aezZim.shape)
print("First few rows:")
print(aezZim.head())

"""

# load political boundaries
# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'

region = "East Africa"


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

# load opt_sw data as dataset!!

model = 'ensemble'

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'
ds = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'
ds = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'
ds = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
ds = xr.open_dataset(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')



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
ds = xr.apply_ufunc(adjust_start,ds)
da = ds['sowing_date']

calendar = calendar_ds['planting_day']
calendar = xr.apply_ufunc(adjust_start, calendar)

# just for East Africa, calculate absolute sowing dates difference
diff_da = xr.full_like(da, fill_value=np.nan)

for lat in range(len(da['lat'])):
    for lon in range(len(da['lon'])):
        print('lat: ' +str(lat) + ' lon: ' + str(lon))
        diff_da[lat,lon] = diff_sw_date(da[lat,lon].values, calendar[lat,lon].values) 

# load yield data
# open saved files
opt_yield_av = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
opt_yield_cv = xr.open_dataset(save_dir+f'/{model}_yieldCV_MgtMais0_optSowingDates_2.0.nc')

cal_yield_av = xr.open_dataset(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')
cal_yield_cv = xr.open_dataset(save_dir+f'/{model}_yieldCV_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

diff_av = xr.open_dataset(save_dir+f'/{model}_Diff_yieldMean_opt_CropCal_SowingDates.nc')
diff_cv = xr.open_dataset(save_dir+f'/{model}_Diff_yieldCV_opt_CropCal_SowingDates.nc')

diff_av_rel = diff_av/opt_yield_av

################### visualizing #####################

# AEZ global SSA for each region
# Define your custom colors
custom_colors = ["white", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                 "#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A133FF"]

# Create a ListedColormap using your custom colors
custom_colormap = ListedColormap(custom_colors)

aez.plot(column ='AEZ', legend=False, cmap=custom_colormap, legend_kwds={'loc': 'lower left', 'fontsize': 7, 'frameon': True, 'title': 'AEZ', 'framealpha': 1})
plt.xlim((-20,55))
plt.ylim((-40,30))
plt.title('Agroecological Zones in SSA', {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/AEZ_SSA_no_legend.png', dpi=260)


# plot optimal sowing dates with global SSA AEZ
fig, ax = plt.subplots(figsize=(10, 8))
#da.plot(cmap='Spectral', vmin=300, vmax=400)
# plt.title(f'Model {model}, {region}: Optimal sowing dates for high and stable maize yield')
calendar.plot(cmap='Spectral', vmin=300, vmax=400)
plt.title(f'{region}: Crop calendar sowing dates')
aez.boundary.plot(ax=ax, linewidth=2, color='green')

"""
gdf_all.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

"""

plt.show()

# plot optimal sowing dates with special Zim AEZ
fig, ax = plt.subplots(figsize=(10, 8))
#da.plot(cmap='Spectral', vmin=300, vmax=400)
# plt.title(f'Model {model}, {region}: Optimal sowing dates for high and stable maize yield')
calendar.plot(cmap='Spectral', vmin=300, vmax=400)
plt.title(f'{region}: Crop calendar sowing dates')
aezZim.boundary.plot(ax=ax, linewidth=2, color='green')

"""
gdf_all.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

"""

plt.show()

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


# clip and visualize the sowing dates for each areas

# boxplot for each SSA AEZ
# with SSA AEZ
reg = aez['AEZ'].unique()
res_sw = pd.DataFrame()

for i in range(len(reg)):
    subset = aez[aez['AEZ'] == reg[i]]
    #da = clip_dset(ds, subset)['sowing_date']
    #da = clip_dset(calendar_ds, subset)['planting_day']
    #da = clip_dset(opt_yield_av, subset)['Yield_average']
    #da = clip_dset(cal_yield_av, subset)['Yield_average']
    #da = clip_dset(diff_av_rel, subset)['Yield_average']
    #da = clip_dset(xr.Dataset({'result_variable':ds['sowing_date']-calendar_ds['planting_day']}), subset)['result_variable']
    da = clip_dset(xr.Dataset({'result_variable':diff_da}), subset)['result_variable']# just for East Africa!!!

    df = da.to_dataframe(name='sowing_date').reset_index()
    res_sw[reg[i]] = df['sowing_date']
  
res_sw.dropna(axis=1, how='all', inplace=True)
#res_sw = res_sw.drop(columns=['No_data'])

res_sw.columns = [col.replace('_', '\n') for col in res_sw.columns]
#res_sw = res_sw*100

plt.title(region, {'fontsize': 15})
plt.xlabel(' ', {'fontsize': 12})
plt.ylabel('Absolute sowing date difference [days]', {'fontsize': 12})  
plt.ylim((0,200))
plt.rc('xtick', labelsize=8)

res_sw.boxplot()
#plt.axhline(0, c='grey', linestyle='dashed')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/sw_diff_AEZ_source_EA.png', dpi=260)


#plt.title(f'{region}: Optimal sowing dates per AEZ')
#plt.title(f'{region}: Crop calendar sowing dates per AEZ')
#plt.title(f'{region}: Yields from optimal sowing dates per AEZ')
#plt.title(f'{region}: Yields from crop calendar sowing dates per AEZ')
#plt.title(f'{region}: Difference of simulated yields per AEZ')
#plt.title(f'{region}: Difference of sowing dates per AEZ')
#plt.title(f'{region}: Absolute difference of sowing dates per AEZ')

#plt.xlabel('AEZ')
#plt.ylabel('Sowing date in doy')
#plt.ylabel('Yield in t/ha')
#plt.ylabel('Yield difference in t/ha')
#plt.ylabel('Sowing date difference in days')

#plt.ylim((0,365))
#plt.ylim((0,13))


# with Zim AEZ
reg = aezZim['Region'].values

for i in range(len(aezZim)):
    plt.figure()
    subset = aezZim[aezZim['Region'] == reg[i]]
    da = clip_dset(ds, subset)['sowing_date']
    da.plot()
    plt.title(f'{reg[i]}')
    plt.xlim((25,34))
    plt.ylim((-23,-15))
    
    
# with SSA AEZ
reg = aez['AEZ'].unique()

for i in range(len(reg)):
    plt.figure()
    subset = aez[aez['AEZ'] == reg[i]]
    da = clip_dset(ds, subset)['sowing_date']
    da.plot(vmin=0, vmax=365)
    plt.title(f'{reg[i]}')
    plt.xlim((28,39))
    plt.ylim((-6,2.5))


# with political region
reg = gdf15['NAME_1'].values


for i in range(len(gdf11)):
    plt.figure()
    subset = gdf15[gdf15['NAME_1'] == reg[i]]
    da = clip_dset(ds, subset)['sowing_date']
    da.plot()
    plt.title(f'{reg[i]}')
    plt.xlim((25,34))
    plt.ylim((-23,-15))


### a boxplot for each shp 
# political region 
reg = gdf11['NAME_1'].values
res_sw = pd.DataFrame()

for i in range(len(gdf11)):
    subset = gdf11[gdf11['NAME_1'] == reg[i]]
    da = clip_dset(ds, subset)['sowing_date']
    df = da.to_dataframe(name='sowing_date').reset_index()
    df['sowing_date']
    res_sw[reg[i]] = df['sowing_date']

res_sw.boxplot()

# for each Zim AEZ
reg = aezZim['Region'].values
res_sw = pd.DataFrame()

for i in range(len(aezZim)):
    subset = aezZim[aezZim['Region'] == reg[i]]
    da = clip_dset(ds, subset)['sowing_date']
    df = da.to_dataframe(name='sowing_date').reset_index()
    df['sowing_date']
    res_sw[reg[i]] = df['sowing_date']

res_sw.boxplot()


