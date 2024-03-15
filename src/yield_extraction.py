# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:27:11 2024

@author: arulrich

get 30 years yield mean and cv simulated with the optimal sowing dates and with thw sowing dates from the GGCMI crop calendar
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def extract_yield(sowing_dates, sim_data, value):
    """Get 30 years yield mean or cv simulated with the optimal sowing dates or with thw sowing dates from the GGCMI crop calendar

    Args:
        sowing_dates (xr.dataarray): map with sowing dates in doy from our approach or from crop calendar
        sim_data (xr.daraarray): map with yield mean or yield cv for every location and sowing date
        value (str): either Yield_average or Yield_CV

    Returns:
        res_da (xr.dataarry): map with yield mean or cv obtained with the given input sowing dates 
    """
    # create empty dataarray to store yield values
    res_da = xr.DataArray(np.nan, coords={'lat': sowing_dates['lat'].values, 'lon': sowing_dates['lon'].values}, dims=['lat','lon'], name = value)  
    
    for i in range(len(res_da['lat'].values)):
        for j in range(len(res_da['lon'].values)):
            print(f'Lat: {i}, Lon: {j}')
            lat_value = res_da['lat'].values[i]
            lon_value = res_da['lon'].values[j]
            sowing_date_value = sowing_dates.sel(lat=lat_value, lon=lon_value, method='nearest').values
            if np.isnan(sowing_date_value):
                res_da.loc[dict(lat=lat_value, lon=lon_value)] = np.nan
            else:
                # Use .interp() to extract the value
                # Interpolate with linear method along 'sowing_date'
                interpolated_data = sim_data.interp(sowing_date=sowing_date_value, method='linear')
                
                # Interpolate with nearest method along 'lat' and 'lon'
                value_at_location = interpolated_data.interp(lat=lat_value, lon=lon_value, method='nearest')

                res_da.loc[dict(lat=lat_value, lon=lon_value)] = value_at_location.values
    
    return res_da


region = 'Zimbabwe'

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'


""" already run it for Zimbabwe, Zambia, WA, and EA

# load data about sowing dates
sw_cal = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zim_crop_calendar_0_05.nc') # crop calendar, res 0.05°
sw_cal = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/WA_crop_calendar_0_05.nc') # crop calendar, res 0.05°
sw_cal = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/EA_crop_calendar_0_05.nc') # crop calendar, res 0.05°
sw_cal = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zam_crop_calendar_0_05.nc') # crop calendar, res 0.05°

sw_opt = xr.open_dataarray(save_dir+'/ensemble_MgtMais0_optSwDateMain_2.0.nc')

# load data about simulations yield mean and cv
cv_yield = xr.open_dataarray(save_dir+'/ensemble_yieldCV_MgtMais0_allSowingDates_2.0.nc')
av_yield = xr.open_dataarray(save_dir+'/ensemble_yieldMean_MgtMais0_allSowingDates_2.0.nc')
av_yield = av_yield.where(av_yield != 0) # set back 0 values to nan

# optimal sowing dates, yield mean
opt_yield_av = extract_yield(sw_opt, av_yield, 'Yield_average')
# optimal sowing dates, yield CV
opt_yield_cv = extract_yield(sw_opt, cv_yield, 'Yield_CV')

# crop calendar sowing dates, yield mean
cal_yield_av = extract_yield(sw_cal, av_yield, 'Yield_average')
# crop calendar sowing dates, yield CV
cal_yield_cv = extract_yield(sw_cal, cv_yield, 'Yield_CV')

# differences
# yield mean: optimal - crop calendar
diff_av = opt_yield_av - cal_yield_av
# yield CV: optimal - crop calendar
diff_cv = opt_yield_cv - cal_yield_cv


# save it
opt_yield_av.to_netcdf(save_dir+'/ensemble_yieldMean_MgtMais0_optSowingDates_2.0.nc')
opt_yield_cv.to_netcdf(save_dir+'/ensemble_yieldCV_MgtMais0_optSowingDates_2.0.nc')

cal_yield_av.to_netcdf(save_dir+'/ensemble_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')
cal_yield_cv.to_netcdf(save_dir+'/ensemble_yieldCV_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

diff_av.to_netcdf(save_dir+'/ensemble_Diff_yieldMean_opt_CropCal_SowingDates.nc')
diff_cv.to_netcdf(save_dir+'/ensemble_Diff_yieldCV_opt_CropCal_SowingDates.nc')
"""

# open saved files
model = 'ensemble'
opt_yield_av = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_optSowingDates_2.0.nc')
opt_yield_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_optSowingDates_2.0.nc')

cal_yield_av = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')
cal_yield_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

diff_av = xr.open_dataarray(save_dir+f'/{model}_Diff_yieldMean_opt_CropCal_SowingDates.nc')
diff_cv = xr.open_dataarray(save_dir+f'/{model}_Diff_yieldCV_opt_CropCal_SowingDates.nc')

# open shp files
# AEZ
aez = gpd.read_file('C:/Users/arulrich/Documents/data_analysis/AEZ/AEZ_global_SSA/aez_named.shp')

# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'


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



# visualize it 
# average yields simulated with optimal simulated sowing dates

opt_yield_av.mean()
opt_yield_av.max()


fig, ax = plt.subplots(figsize=(10, 8))
opt_yield_av.plot(vmin=0, vmax=13, cmap='plasma')
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='paleturquoise')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/opt_yield_EA.png', dpi=260)


# average yields simulated with crop calendar sowing dates

cal_yield_av.mean()
cal_yield_av.max()

fig, ax = plt.subplots(figsize=(10, 8))
cal_yield_av.plot(vmin=0, vmax=13, cmap='plasma')
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='paleturquoise')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Calendar_yield_EA.png', dpi=260)


# CV of yields simulated with optimal sowing dates
df = opt_yield_cv.to_dataframe(name='CV').reset_index()
plt.hist(df['CV'], bins=36)

fig, ax = plt.subplots(figsize=(10, 8))
opt_yield_cv.plot(vmin=0, vmax=1.5, cmap='Wistia')
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')
gdf_all.boundary.plot(ax=ax, linewidth=1, color='paleturquoise')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')


# CV of  yields simulated with crop calendar sowing dates
df = cal_yield_cv.to_dataframe(name='CV').reset_index()
plt.hist(df['CV'], bins=36)

fig, ax = plt.subplots(figsize=(10, 8))
cal_yield_cv.plot(vmin=0, vmax=1.5, cmap='Wistia')
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')
gdf_all.boundary.plot(ax=ax, linewidth=1, color='paleturquoise')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')


# visualize diff
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# difference in averaged yields

diff_av.min()
diff_av.max()
diff_av.mean()
diff_av.quantile(0.25)
diff_av.quantile(0.75)
diff_av.std()

df = diff_av.to_dataframe(name='Diff_Yield_average').reset_index()
plt.hist(df['Diff_Yield_average'], bins=36)
plt.title('Distribution of yields differences')
plt.xlabel('Difference in t pro ha')
plt.ylabel('Amount of pixel')

norm = MidpointNormalize(midpoint=0, vmin=-4, vmax=8)

fig, ax = plt.subplots(figsize=(10, 8))
diff_av.plot(cmap='coolwarm', norm=norm)
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='cyan')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/yield_diff_opt_cal_EA.png', dpi=260)



# difference in CV of yields
diff_cv.min()
diff_cv.max()
diff_cv.quantile(0.05)
diff_cv.quantile(0.95)

df = diff_cv.to_dataframe(name='Diff_Yield_CV').reset_index()
plt.hist(df['Diff_Yield_CV'], bins=36)
plt.title('Distribution of CV differences')
plt.xlabel('')
plt.ylabel('Amount of pixel')

norm = MidpointNormalize(midpoint=0, vmin=-0.7, vmax=1)

fig, ax = plt.subplots(figsize=(10, 8))
diff_cv.plot(cmap='coolwarm', norm=norm)

plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
#aez.boundary.plot(ax=ax, linewidth=2, color='green')

gdf_all.boundary.plot(ax=ax, linewidth=1, color='paleturquoise')
gdf_all_nat.boundary.plot(ax=ax, linewidth=2, color='black')

##################### regression averaged yield values ########################

# extract variables' values
df_opt = opt_yield_av.to_dataframe(name='Yield').reset_index()
df_cal  = cal_yield_av.to_dataframe(name='Yield_cal').reset_index()

# add coordinate column as id
df_opt['coord'] = np.nan
for i in range(len(df_opt['lat'])):
    print(i)
    df_opt['coord'][i] = str(df_opt['lat'][i])+'_' + str(df_opt['lon'][i])

df_cal['coord'] = np.nan
for i in range(len(df_cal['lat'])):
    print(i)
    df_cal['coord'][i] = str(df_cal['lat'][i])+'_' + str(df_cal['lon'][i])
    
# join the dfs
reg_df = pd.merge(df_opt, df_cal.loc[:,['Yield_cal', 'coord']], on='coord')


reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('Yield ~ Yield_cal', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['Yield_cal'], reg_df['Yield'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['Yield_cal'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['Yield_cal'], iv_u, "r--", label='upper and lower conf. interval' )
plt.plot(reg_df['Yield_cal'], iv_l, "r--")
plt.xlim((0,13))
plt.ylim((0,13))
plt.xlabel('Yields from crop calendar: t/ha')
plt.ylabel('Yields from optimal sowing dates: t/ha')
plt.title('Regression: yields from optimal vs crop calendar')
plt.text(10, 7, f'R2 = {round(res.rsquared,3)}')
plt.text(10, 5.5, f'p value = {round(res.f_pvalue,4)}')
plt.plot([0,13], [0,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.legend(loc="best")
