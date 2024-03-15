# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:08:01 2023

@author: arulrich

comparison between crop calendar and our optimal sowing dates
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# chose directory

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'

# load data
# optimal simulated sowing dates
model = 'ensemble'
da = xr.open_dataarray(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.nc')

# crop calendar
ggcmi = xr.open_dataset('C:/Users/arulrich/Desktop/MA/Download_data/Crop_calender/mai_rf_ggcmi_crop_calendar_phase3_v1.01.nc4')
da_ggcmi = ggcmi['planting_day']


def conv_ggcmi(ggcmi_array, da, new_crs = "EPSG:4326"):
    """Regrid the crop calendar so that it has the same extent, resolution and crs as da

    Args:
        ggcmi_array (xr.Dataarray): Dataarray containing the sowing_dates from the crop calendar of ggcmi
        da (xr.Dataarray): Dataarray containing the optimal sowing dates
        new_crs (str): coordinate system to use, default is "EPSG:4326" # EPSG code for WGS 84
        maturation_time (int): The minimum time required between planting and harvest for the specific crop
        ranking (int): The minimum number of sowing dates to consider in a sowing window

    Returns:
        resampled_variable (xr.Dataarray): Dataarray of crop calendar regridded.
    """
    new_latitudes = da['lat']
    new_longitudes = da['lon']
    
    # Resample the variable to the new resolution, extent and coordinate system
    resampled_variable = ggcmi_array.interp(lon=new_longitudes,lat=new_latitudes,  method='nearest')
    resampled_variable = resampled_variable.assign_coords(crs=new_crs)
    return resampled_variable


"""
# Regrid crop calendar and save it
# already run it 

calendar = conv_ggcmi(da_ggcmi, da)
calendar.to_netcdf('C:/Users/arulrich/Documents/data_analysis/Zam_crop_calendar_0_05.nc')
"""


# load the right calendar according to the region
calendar = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zim_crop_calendar_0_05.nc') # Zimbabwe
calendar = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/EA_crop_calendar_0_05.nc') # East Africa
calendar = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/WA_crop_calendar_0_05.nc') # West Africa
calendar = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zam_crop_calendar_0_05.nc') # Zambia



###################### Visualize ############################################
# plot the calendar in the same way (same parameters) as the optimal sowing dates plots

# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'

region = 'East Africa'

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


# open shp of AEZ
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')


calendar.min()
calendar.max()
calendar.quantile(0.05)
calendar.quantile(0.95)
calendar.quantile(0.25)
calendar.quantile(0.75)
calendar.mean()

df = calendar.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=36)
plt.ylim((0,1))


# adjust start of the season for the entire area
start_season = 250 # 250 for Zimbabwe, 170 for Zambia, 0 for EA and WA

def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x

calendar = xr.apply_ufunc(adjust_start,calendar)
da = xr.apply_ufunc(adjust_start, da)

df = da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=36)
plt.ylim((0,20))

da.min()
da.max()
da.quantile(0.05)
da.quantile(0.95)

calendar.min()
calendar.max()
calendar.quantile(0.05)
calendar.quantile(0.95)

df = calendar.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=36)


fig, ax = plt.subplots(figsize=(10, 8))

calendar.plot(cmap='gist_rainbow', vmin =0, vmax=365)
plt.xlim((calendar['lon'][0], calendar['lon'][-1]))
plt.ylim((calendar['lat'][0], calendar['lat'][-1]))

plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='lightgray')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Calendar_EA.png', dpi=260)

plt.show()



##########################  absolute difference ##############################
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

diff_da = xr.full_like(da, fill_value=np.nan)

for lat in range(len(da['lat'])):
    for lon in range(len(da['lon'])):
        print('lat: ' +str(lat) + ' lon: ' + str(lon))
        diff_da[lat,lon] = diff_sw_date(da[lat,lon].values, calendar[lat,lon].values)    


diff_da.min()
diff_da.max()
diff_da.mean()
diff_da.std()


diff_da.quantile(0.25)
diff_da.quantile(0.75)

df = diff_da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=30)
plt.title('Distribution of sowing dates differences')
plt.xlabel('Difference in doy')
plt.ylabel('Amount of pixel')

fig, ax = plt.subplots(figsize=(10, 8))

diff_da.plot(cmap='Reds', vmin =0, vmax=180)

plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})

#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/sw_abs_diff_opt_cal_EA.png', dpi=260)


plt.show()


##########################  difference optimal - crop calendar ##############################

da_adj = xr.apply_ufunc(adjust_start,da)
diff_da = da_adj - calendar


diff_da.min()
diff_da.max()
diff_da.mean()
diff_da.std()


diff_da.quantile(0.25)
diff_da.quantile(0.75)

df = diff_da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=32)
plt.title('Distribution of sowing dates differences')
plt.xlabel('Difference in doy')
plt.ylabel('Amount of pixel')

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

norm = MidpointNormalize(midpoint=0, vmin=-150, vmax=150)

fig, ax = plt.subplots(figsize=(10, 8))

diff_da.plot(cmap='coolwarm',norm=norm)

plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})

#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/sw_diff_opt_cal_WA.png', dpi=260)


plt.show()



################################## regression  ################################
# extract variables' values
df_adj = da_adj.to_dataframe(name='sowing_date').reset_index()
df_cal  = calendar.to_dataframe(name='sowing_date_cal').reset_index()

# add coordinate column as id
df_adj['coord'] = np.nan
for i in range(len(df_adj['lat'])):
    print(i)
    df_adj['coord'][i] = str(df_adj['lat'][i])+'_' + str(df_adj['lon'][i])

df_cal['coord'] = np.nan
for i in range(len(df_cal['lat'])):
    print(i)
    df_cal['coord'][i] = str(df_cal['lat'][i])+'_' + str(df_cal['lon'][i])
    
# join the dfs
reg_df = pd.merge(df_adj, df_cal.loc[:,['sowing_date_cal', 'coord']], on='coord')


reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('sowing_date ~ sowing_date_cal', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sowing_date_cal'], reg_df['sowing_date'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sowing_date_cal'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sowing_date_cal'], iv_u, "r--", label='upper and lower conf. interval' )
plt.plot(reg_df['sowing_date_cal'], iv_l, "r--")
plt.xlim((-100,420))
plt.ylim((-100,420))
plt.xlabel('Sowing dates crop calendar: doy')
plt.ylabel('Optimal sowing dates: doy')
plt.title('Regression: optimal vs crop calendar sowing dates')
plt.text(150, 130, f'R2 = {round(res.rsquared,3)}')
plt.text(150, 110, f'p value = {round(res.f_pvalue,4)}')
plt.plot([-100,420], [-100,420], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.legend(loc="best")




########################### same analysis but with arrays with crop calendar resolution (to do better)
"""
# da with resolution of crop calendar 
new_resolution = -da_ggcmi['lat'][1]+ da_ggcmi['lat'][0]  # degrees
new_crs = "EPSG:4326"  # EPSG code for WGS 84
start_season = 250

def conv_da(ggcmi_array, da, new_res, new_crs, start_season, method, coarsen_factor):
    ggcmi_array = xr.apply_ufunc(adjust_start,ggcmi_array)
    da = xr.apply_ufunc(adjust_start,da)
    if method == 'median':
        da = da.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').median()
    elif method == 'mean':
        da = da.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').median()

    new_latitudes = np.arange(da['lat'][0], da['lat'][len(da['lat'])-1] + new_res, new_res)
    new_longitudes = np.arange(da['lon'][0], da['lon'][len(da['lon'])-1] + new_res, new_res)
    
    # Resample the variable to the new resolution, extent, and coordinate system
    resampled_variable_cal = ggcmi_array.interp(lon=new_longitudes,lat=new_latitudes,  method='nearest')
    resampled_variable_cal = resampled_variable_cal.assign_coords(crs=new_crs)
    
    return resampled_variable_cal, da

calendar_0_5, da_0_5 = conv_da(da_ggcmi, da, new_resolution, new_crs, start_season, 'median', 10)
calendar_0_5.to_netcdf('C:/Users/arulrich/Documents/data_analysis/Zim_crop_calendar_0_5.nc')
da_0_5.to_netcdf('C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output/ensemble_MgtMais0_optSwDateMain_2.0_res_0_5.nc')

"""
calendar_0_5 = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Zim_crop_calendar_0_5.nc')
da_0_5 = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output/ensemble_MgtMais0_optSwDateMain_2.0_res_0_5.nc')


da_0_5.plot(vmin=300, vmax=400, cmap='Spectral')
calendar_0_5.plot(vmin=300, vmax=400, cmap='Spectral')

###################### Visualize ############################################
# plot the calendar in the same way (same parameters) as the optimal sowing dates plots

# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp')
gdf02 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_0.shp')
gdf03 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp')
gdf04 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_0.shp')
gdf05 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_0.shp')

gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')
gdf12 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_1.shp')
gdf13 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')
gdf14 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_1.shp')
gdf15 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_1.shp')


calendar_0_5.min()
calendar_0_5.max()
calendar_0_5.quantile(0.05)
calendar_0_5.quantile(0.95)

df = calendar_0_5.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=36)

fig, ax = plt.subplots(figsize=(10, 8))
calendar_0_5.plot(cmap='Spectral', vmin =300, vmax=400)
plt.title('Zimbabwe: Crop calendar of GGCMI')
#da_0_5.plot(vmin=300, vmax=400, cmap='Spectral')
#plt.title('Model ensemble, Zimbabwe:Optimal sowing dates for high and stable maize yield')
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf12.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf13.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf14.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf15.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf01.boundary.plot(ax=ax, linewidth=2, color='black')
gdf02.boundary.plot(ax=ax, linewidth=2, color='black')
gdf03.boundary.plot(ax=ax, linewidth=2, color='black')
gdf04.boundary.plot(ax=ax, linewidth=2, color='black')
gdf05.boundary.plot(ax=ax, linewidth=2, color='black')

plt.show()


##########################  absolute difference ##############################
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

diff_da = xr.full_like(da_0_5, fill_value=np.nan)

for lat in range(len(da_0_5['lat'])):
    for lon in range(len(da_0_5['lon'])):
        print('lat: ' +str(lat) + ' lon: ' + str(lon))
        diff_da[lat,lon] = diff_sw_date(da_0_5[lat,lon].values, calendar_0_5[lat,lon].values)    


diff_da.min()
diff_da.max()
diff_da.quantile(0.05)
diff_da.quantile(0.95)

df = diff_da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'], bins=30)
plt.title('Distribution of sowing dates differences')
plt.xlabel('Difference in doy')
plt.ylabel('Amount of pixel')

fig, ax = plt.subplots(figsize=(10, 8))
diff_da.plot(cmap='Spectral', vmin =0, vmax=75)
plt.title('Absolute difference: Crop calendar vs Optimal Sowing dates')
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf12.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf13.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf14.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf15.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf01.boundary.plot(ax=ax, linewidth=2, color='black')
gdf02.boundary.plot(ax=ax, linewidth=2, color='black')
gdf03.boundary.plot(ax=ax, linewidth=2, color='black')
gdf04.boundary.plot(ax=ax, linewidth=2, color='black')
gdf05.boundary.plot(ax=ax, linewidth=2, color='black')

plt.show()


##########################  difference optimal - crop calendar ##############################

diff_da = da_0_5 - calendar_0_5


diff_da.min()
diff_da.max()
diff_da.quantile(0.05)
diff_da.quantile(0.95)

df = diff_da.to_dataframe(name='sowing_date').reset_index()
plt.hist(df['sowing_date'])#, bins=32, range=[-60,260])
plt.title('Distribution of sowing dates differences')
plt.xlabel('Difference in doy')
plt.ylabel('Amount of pixel')


norm = MidpointNormalize( midpoint = 0, vmin=-20, vmax=50)


fig, ax = plt.subplots(figsize=(10, 8))
diff_da.plot(cmap='coolwarm', norm=norm)
plt.title('Difference: Optimal Sowing dates - Crop calendar')
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf12.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf13.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf14.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf15.boundary.plot(ax=ax, linewidth=1, color='blue')

gdf01.boundary.plot(ax=ax, linewidth=2, color='black')
gdf02.boundary.plot(ax=ax, linewidth=2, color='black')
gdf03.boundary.plot(ax=ax, linewidth=2, color='black')
gdf04.boundary.plot(ax=ax, linewidth=2, color='black')
gdf05.boundary.plot(ax=ax, linewidth=2, color='black')

plt.show()


################################## regression  ################################
# extract variables' values
df_0_5 = da_0_5.to_dataframe(name='sowing_date').reset_index()
df_cal_0_5  = calendar_0_5.to_dataframe(name='sowing_date_cal').reset_index()

# add coordinate column as id
df_0_5['coord'] = np.nan
for i in range(len(df_0_5['lat'])):
    print(i)
    df_0_5['coord'][i] = str(df_0_5['lat'][i])+'_' + str(df_0_5['lon'][i])

df_cal_0_5['coord'] = np.nan
for i in range(len(df_cal_0_5['lat'])):
    print(i)
    df_cal_0_5['coord'][i] = str(df_cal_0_5['lat'][i])+'_' + str(df_cal_0_5['lon'][i])
    
# join the dfs
reg_df = pd.merge(df_0_5, df_cal_0_5.loc[:,['sowing_date_cal', 'coord']], on='coord')

reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('sowing_date ~ sowing_date_cal', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sowing_date_cal'], reg_df['sowing_date'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sowing_date_cal'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sowing_date_cal'], iv_u, "r--", label='upper and lower conf. interval' )
plt.plot(reg_df['sowing_date_cal'], iv_l, "r--")
plt.xlim((280,500))
plt.ylim((280,500))
plt.xlabel('Sowing dates crop calendar: doy')
plt.ylabel('Optimal sowing dates: doy')
plt.title('Regression: optimal vs crop calendar sowing dates')
plt.text(450, 370, f'R2 = {round(res.rsquared,3)}')
plt.text(450, 350, f'p value = {round(res.f_pvalue,4)}')
plt.plot([280, 500], [280,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.legend(loc="best")

