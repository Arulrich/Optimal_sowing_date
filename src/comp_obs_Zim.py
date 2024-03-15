# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:20:19 2023

@author: arulrich
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


""" THIS PROCESS ALREADY DONE

# load data
obs = pd.read_csv('C:/Users/arulrich/Desktop/MA/survey/planting_dates_zim_2021.csv')
obs.columns
obs.shape
obs.size
obs.head
obs.describe()
obs.info()

obs['planting_date_dekad'].unique()

#### create a column with the sowing dates in doy
obs['sw_doy'] = obs['lat'] 
obs['sw_doy'] = float('nan')

for i in range(len(obs['planting_date_dekad'])):
    if obs['planting_date_dekad'][i] == '1/10': # 5th of October
        obs['sw_doy'][i] = 278
    elif obs['planting_date_dekad'][i] == '2/10':
        obs['sw_doy'][i] = 288
    elif obs['planting_date_dekad'][i] == '3/10':
        obs['sw_doy'][i] = 298
    elif obs['planting_date_dekad'][i] == '1/11':
        obs['sw_doy'][i] = 309
    elif obs['planting_date_dekad'][i] == '2/11':
        obs['sw_doy'][i] = 319
    elif obs['planting_date_dekad'][i] == '3/11':
        obs['sw_doy'][i] = 329
    elif obs['planting_date_dekad'][i] == '1/12':
        obs['sw_doy'][i] = 339
    elif obs['planting_date_dekad'][i] == '2/12':
        obs['sw_doy'][i] = 349
    elif obs['planting_date_dekad'][i] == '3/12':
        obs['sw_doy'][i] = 359
    elif obs['planting_date_dekad'][i] == '1/1':
        obs['sw_doy'][i] = 370
    elif obs['planting_date_dekad'][i] == '2/1':
        obs['sw_doy'][i] = 380
    elif obs['planting_date_dekad'][i] == '3/1':
        obs['sw_doy'][i] = 390

plt.hist(obs['sw_doy'], bins=25, range=(250,500))
plt.title('Distribution of sowing dates')
plt.xlabel('doy')
plt.ylabel('Amount of locations')
plt.xlim((250,500))

###################### create a column with values of optimal sowing dates
obs['opt_sw'] = obs['lat']
obs['opt_sw'] = float('nan')

# load data
# optimal sowing dates
model = 'ensemble'
da = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output/{model}_MgtMais0_optSwDateMain_2.0.nc')

for i in range(len(obs['planting_date_dekad'])):
    obs['opt_sw'][i] = da.sel(lat = obs['lat'][i], lon = obs['lon'][i], method='nearest').values
    
start_season = 250
import numpy as np
def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x
obs['opt_sw'] = obs['opt_sw'].apply(adjust_start)

obs['opt_sw'].describe()
# how many nan values
obs['opt_sw'].isna().sum()


plt.hist(obs['opt_sw'], bins=25, range=(250,500))
plt.title('Distribution of optimal sowing dates')
plt.xlabel('doy')
plt.ylabel('Amount of locations')
plt.xlim((250,500))


###################### create a column with absolute difference between obs and simulated sowing date
obs['abs_diff'] = ((obs['opt_sw'] - obs['sw_doy'])**2)**(1/2)

obs['abs_diff'].describe()


plt.hist(obs['abs_diff'], bins=27, range=(0,160))
plt.title('Distribution of absolute difference between sowing dates')
plt.xlabel('doy')
plt.ylabel('Amount of locations')

###################### create a column with difference between obs and simulated sowing date
obs['diff'] = obs['opt_sw'] - obs['sw_doy']

obs['diff'].describe()

plt.hist(obs['diff'], bins=24, range=(-80,160))
plt.title('Distribution of difference between sowing dates: optimal - observed')
plt.xlabel('doy')
plt.ylabel('Amount of locations')

####################### create a column with values of sowing date from crop calendar
obs['calend'] = obs['lat']
obs['calend'] = float('nan')

# load data
# crop calendar
da = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Zim_crop_calendar_0_05.nc')

for i in range(len(obs['planting_date_dekad'])):
    obs['calend'][i] = da.sel(lat = obs['lat'][i], lon = obs['lon'][i], method='nearest').values

    
start_season = 250
def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x
obs['calend'] = obs['calend'].apply(adjust_start)

obs['calend'].describe()
# how many nan values
obs['calend'].isna().sum()


plt.hist(obs['calend'], bins=38, range=(240,620))
plt.title('Distribution of sowing dates in Crop Calendar')
plt.xlabel('doy')
plt.ylabel('Amount of locations')
plt.xlim((240,620))


######################### create a column with absolute difference between obs and crop calendar sowing date
obs['abs_diff_cal'] = ((obs['calend'] - obs['sw_doy'])**2)**(1/2)

obs['abs_diff_cal'].describe()


plt.hist(obs['abs_diff_cal'], bins=9, range=(0,90), histtype='stepfilled')
plt.title('Distribution of absolute difference between sowing dates: crop calendar - observed')
plt.xlabel('doy')
plt.ylabel('Amount of locations')


######################### create a column with absolute difference between obs and crop calendar sowing date
obs['diff_cal'] = obs['calend'] - obs['sw_doy']

obs['diff_cal'].describe()


plt.hist(obs['diff_cal'], bins=37, range=(-100,270))
plt.title('Distribution of difference between sowing dates: crop calendar - observed')
plt.xlabel('doy')
plt.ylabel('Amount of locations')



# add simulated yield given from the sowing date
def extract_yearly_yield(dataarray, year, sowing_date, lat, lon):
    year_data = dataarray.sel(time=year)

    # Interpolate with linear method along 'sowing_date'
    interpolated_data = year_data.interp(sowing_date=sowing_date, method='linear')
    
    # Interpolate with nearest method along 'lat' and 'lon'
    value_at_location = interpolated_data.interp(lat=lat, lon=lon, method='nearest')
    
    return value_at_location

# load dataarray with the simulated yield values for each sowing date, year, and location (we need sowing date of the growing season 2021/22)
model = 'ensemble'
dataarray = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_90_2020_2022/output/{model}_yield_MgtMais0_allSowingDates_2.0.nc')
dataarray = dataarray.where(dataarray != 0) # set 0 values back to Nan

# add column about the year of the sowing date

obs['sim_yield_obs_22'] = np.nan
obs['sim_yield_opt_22'] = np.nan
obs['sim_yield_cal_22'] = np.nan

# for survey observation 
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['opt_sw'][i]):
        obs['sim_yield_obs_22'][i] = np.nan
    else:
        if obs['sw_doy'][i] > 365:
            year = 2022
        else:
            year = 2021
        obs['sim_yield_obs_22'][i] = extract_yearly_yield(dataarray=dataarray, year=year, sowing_date = obs['sw_doy'][i], lat= obs['lat'][i], lon= obs['lon'][i])

plt.hist(obs['sim_yield_obs_22'], bins=200)
plt.xlim((0,14))
plt.ylim((0,400))
obs['sim_yield_obs_22'].describe()

# for optimal sowing dates
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['opt_sw'][i]):
        obs['sim_yield_opt_22'][i] = np.nan
    else:
        if obs['opt_sw'][i] > 365:
            year = 2022
        else:
            year = 2021
        obs['sim_yield_opt_22'][i] = extract_yearly_yield(dataarray=dataarray, year=year, sowing_date = obs['opt_sw'][i], lat= obs['lat'][i], lon= obs['lon'][i])

plt.hist(obs['sim_yield_opt_22'], bins=200)
plt.xlim((0,14))
plt.ylim((0,450))
obs['sim_yield_opt_22'].describe()

# for crop calendar sowing dates
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['opt_sw'][i]):
        obs['sim_yield_cal_22'][i] = np.nan
    else:
        if obs['calend'][i] > 365:
            year = 2022
        else:
            year = 2021
        obs['sim_yield_cal_22'][i] = extract_yearly_yield(dataarray=dataarray, year=year, sowing_date = obs['calend'][i], lat= obs['lat'][i], lon= obs['lon'][i])


plt.hist(obs['sim_yield_cal_22'], bins=200)
plt.xlim((0,14))
plt.ylim((0,450))
obs['sim_yield_cal_22'].describe()

# difference
obs['diff_yield'] = obs['sim_yield_opt_22'] - obs['sim_yield_obs_22']
obs['diff_yield_cal'] = obs['sim_yield_cal_22'] - obs['sim_yield_obs_22']
obs['diff_yield_opt_cal'] = obs['sim_yield_opt_22'] - obs['sim_yield_cal_22']

######################## save resp. load the dataframe containing the data of the survey and from our interpolations
obs.to_csv('C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/planting_dates_zim_2021_and_our_data.csv')
"""
# load it
obs = pd.read_csv('C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/planting_dates_zim_2021_and_our_data.csv')
obs = obs.iloc[:,1:]



####################### regression observed vs simulated
# Fit regression model
# Create the df 
reg_df = obs[['opt_sw', 'sw_doy', 'sector']]
reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('opt_sw ~ sw_doy', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sw_doy'], reg_df['opt_sw'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sw_doy'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sw_doy'], iv_u, "r--", label='upper and lower conf. interval')
plt.plot(reg_df['sw_doy'], iv_l, "r--")
plt.plot([270,500], [270,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((270,500))
plt.ylim((270,500))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Optimal sowing dates: doy')
plt.title('Regression: optimal vs observed sowing dates')
plt.text(460, 380, f'R2 = {round(res.rsquared,3)}')
plt.text(460, 360, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = reg_df[reg_df['sector'] == typeFarm]
    model = smf.ols('opt_sw ~ sw_doy', data=plot_df)
    res = model.fit()
    # Inspect the results
    print(f'{typeFarm}')
    print(res.summary())
    
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    
    plt.plot(plot_df['sw_doy'], plot_df['opt_sw'], "o", label='data')
    plt.plot(plot_df['sw_doy'], res.fittedvalues, "b--.", label="OLS")
    plt.plot(plot_df['sw_doy'], iv_u, "r--", label='upper and lower conf. interval')
    plt.plot(plot_df['sw_doy'], iv_l, "r--")
    plt.plot([270,500], [270,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
    plt.xlim((270,500))
    plt.ylim((270,500))
    plt.xlabel('Observed sowing dates: doy')
    plt.ylabel('Optimal sowing dates: doy')
    plt.title(f'Regression: optimal vs observed sowing dates, farm type: {typeFarm}')
    plt.text(450, 380, f'R2 = {round(res.rsquared,3)}')
    plt.text(450, 360, f'p value = {round(res.f_pvalue,4)}')
    plt.legend(loc="best")

## one plot of them differentiating communal vs commercial areas
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = reg_df[reg_df['sector'] == 'CA']
    else:
        plot_df = reg_df[reg_df['sector'] != 'CA']
    
    model = smf.ols('opt_sw ~ sw_doy', data=plot_df)
    res = model.fit()
    # Inspect the results
    print(f'{typeFarm}')
    print(res.summary())
    
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    
    plt.plot(plot_df['sw_doy'], plot_df['opt_sw'], "o", label='data')
    plt.plot(plot_df['sw_doy'], res.fittedvalues, "b--.", label="OLS")
    plt.plot(plot_df['sw_doy'], iv_u, "r--", label='upper and lower conf. interval')
    plt.plot(plot_df['sw_doy'], iv_l, "r--")
    plt.plot([270,500], [270,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
    plt.xlim((270,500))
    plt.ylim((270,500))
    plt.xlabel('Observed sowing dates: doy')
    plt.ylabel('Optimal sowing dates: doy')
    plt.title(f'Regression: optimal vs observed sowing dates in {typeFarm} farms')
    plt.text(450, 380, f'R2 = {round(res.rsquared,3)}')
    plt.text(450, 360, f'p value = {round(res.f_pvalue,4)}')
    plt.legend(loc="best")



####################### visualize where are the location and our crop mask
model = 'ensemble'
da = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output/{model}_MgtMais0_optSwDateMain_2.0.nc')


dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp')
gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')

mask = da.fillna(0)
mask = mask.where(mask == 0, other=1) # give the value 1 to the pixels which not fullfill the condition == 0
mask = mask.where(mask != 0) # give the value nan to the pixels which not fullfill the condition != 0
cmap = mcolors.ListedColormap(['white', 'green'])
obs_no_CA = obs[obs['sector']!='CA'] # commercial farmers
obs_CA = obs[obs['sector']=='CA'] # communal areas

fig, ax = plt.subplots(figsize=(10, 8))

plt.plot(obs['lon'], obs['lat'], marker = '+', c = 'red', linestyle = '', markersize = 0.5, label = 'Observed farms') # all the farms

#plt.plot(obs_no_CA['lon'], obs_no_CA['lat'], marker = '+', c = 'yellow', linestyle = '', markersize = 0.5, label = 'observed commercial farms')
#plt.plot(obs_CA['lon'], obs_CA['lat'], marker = '+', c = 'pink', linestyle = '', markersize = 0.5, label = 'observed farms in communal areas')
mask.plot(cmap=cmap, label='Crop mask')
plt.title('Zimbabwe: Observed sowing dates and crop mask', {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')
plt.legend(loc='best', markerscale = 11)

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Obs_crop_mask_Zim.png', dpi=260)



######################## regression observed vs crop calendar
# Fit regression model
# Create the df 
reg_df = obs[['calend', 'sw_doy']]
reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('calend ~ sw_doy', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sw_doy'], reg_df['calend'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sw_doy'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sw_doy'], iv_u, "r--", label='upper and lower conf. interval' )
plt.plot(reg_df['sw_doy'], iv_l, "r--")
plt.plot([270,400], [270,400], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((270,400))
plt.ylim((270,400))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Crop calendar sowing dates: doy')
plt.title('Regression: crop calendar vs observed sowing dates')
plt.text(340, 390, f'R2 = {round(res.rsquared,3)}')
plt.text(340, 380, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")


###################### compare differences diff_calend and diff
comp_diff = obs[['diff_cal', 'diff']]
comp_diff.dropna(inplace=True)
# plt.violinplot([comp_diff['diff_cal'], comp_diff['diff']], showmeans=True, showmedians=True)
plt.boxplot([comp_diff['diff_cal'], comp_diff['diff']], labels=['Crop Calendar', 'Optimal'] )
plt.axhline(0, c='grey', linestyle='dashed')
plt.title("Difference between observed and optimal, resp. crop calendar's sowing dates")
plt.ylabel('days')
#plt.ylim((-90,50))

###################### visualize observed, crop calendar and optimal sowing dates distribution with distribution line
# Create the df 
df_dist = obs[['calend', 'sw_doy', 'opt_sw']]#, 'sector']]
df_dist.dropna(inplace=True)

df_dist['sw_doy'].describe()
df_dist['opt_sw'].describe()
df_dist['calend'].describe()

fig,ax = plt.subplots(figsize=(8, 6))

sns.kdeplot(df_dist['calend'], label = "Crop Calendar", fill = True, bw_adjust=1.5) # bw_adjust ==> to have the line smoother
sns.kdeplot(df_dist['opt_sw'], label = "Optimal Simulated",  fill = True, bw_adjust=1.5)
sns.kdeplot(df_dist['sw_doy'], label = "Observation",  fill = True, bw_adjust=1.5)

plt.xlim((290,400))
plt.ylim((0,0.28))
plt.xlabel('Sowing date [doy]', {'fontsize': 12})
plt.ylabel('Density', {'fontsize': 12})
plt.title('Zimbabwe: Distribution of sowing dates', {'fontsize': 15})
plt.legend(loc='best')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Distrib_Obs_Zim.png', dpi=260)



## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = df_dist[df_dist['sector'] == typeFarm]
    sns.kdeplot(plot_df['calend'], label = "Crop Calendar", fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['sw_doy'], label = "Observation",  fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['opt_sw'], label = "Optimal sowing dates",  fill = True, bw_adjust=2)
    plt.xlim((290,400))
    plt.ylim((0,0.22))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, farm type: {typeFarm}')
    plt.legend(loc='best')

## one plot for commercial and one for communal farms
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = df_dist[df_dist['sector'] == 'CA']
    else:
        plot_df = df_dist[df_dist['sector'] != 'CA']
    sns.kdeplot(plot_df['calend'], label = "Crop Calendar", fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['sw_doy'], label = "Observation",  fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['opt_sw'], label = "Optimal sowing dates",  fill = True, bw_adjust=2)    
    plt.xlim((290,400))
    plt.ylim((0,0.22))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates in {typeFarm} farms')
    plt.legend(loc='best')

## distribution of sowing dates depending on the farmtype and on the datasource
for j in range(0,3):
    datasource = ['calend', 'opt_sw', 'sw_doy'][j]
    fig,ax = plt.subplots(figsize=(8, 6))
    for i in range(0,5):
        typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
        plot_df = df_dist[df_dist['sector'] == typeFarm]
        sns.kdeplot(plot_df[datasource], label = f'{typeFarm}',  fill = True, bw_adjust=2)
    plt.xlim((290,400))
    plt.ylim((0,0.19))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, {datasource}')
    plt.legend(loc='best')

## distribution of sowing dates depending on the farmtype (communal vs commercial) and on the datasource
for j in range(0,3):
    datasource = ['calend', 'opt_sw', 'sw_doy'][j]
    fig,ax = plt.subplots(figsize=(8, 6))
    for i in range(0,2):
        typeFarm = ['communal', 'commercial'][i]
        if typeFarm == 'communal':
            plot_df = df_dist[df_dist['sector'] == 'CA']
        else:
            plot_df = df_dist[df_dist['sector'] != 'CA']
        sns.kdeplot(plot_df[datasource], label = f'{typeFarm}',  fill = True, bw_adjust=2)
    plt.xlim((290,400))
    plt.ylim((0,0.19))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, {datasource}')
    plt.legend(loc='best')

###################### visualize observed, crop calendar and optimal sowing dates distribution with histograms (to do better)
# Create the df 
df_dist = obs[['calend', 'sw_doy', 'opt_sw', 'sector']]
df_dist.dropna(inplace=True)

for i in range(0,11):
    fig,ax = plt.subplots(figsize=(8, 6))
    alpha = np.arange(0,1.1,0.1)[i]
    plt.hist(df_dist['sw_doy'], bins=33, range=(290,400), histtype='bar', label = "Observation", alpha = alpha, color='blue')
    plt.hist(df_dist['opt_sw'], bins=33, range=(290,400), histtype='bar', label = "Optimal sowing dates", alpha = alpha, color='magenta')
    plt.hist(df_dist['calend'], bins=33, range=(290,400), histtype='bar', label = "Crop Calendar", alpha = alpha, color='yellow')

    plt.suptitle(f'Alpha = {alpha}')
    plt.xlim((290,400))
    plt.ylim((0,28000))
    plt.xlabel('Sowing date: doy')
    plt.title('Distribution of sowing dates')
    plt.legend(loc='best')


## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = df_dist[df_dist['sector'] == typeFarm]
    sns.distplot(plot_df['calend'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Crop Calendar")
    sns.distplot(plot_df['sw_doy'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Observation")
    sns.distplot(plot_df['opt_sw'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Optimal sowing dates")
    plt.xlim((290,400))
    plt.ylim((0,0.5))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, farm type: {typeFarm}')
    plt.legend(loc='best')

## one plot for commercial and one for communal farms
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = df_dist[df_dist['sector'] == 'CA']
    else:
        plot_df = df_dist[df_dist['sector'] != 'CA']
        
    sns.distplot(plot_df['calend'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Crop Calendar")
    sns.distplot(plot_df['sw_doy'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Observation")
    sns.distplot(plot_df['opt_sw'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Optimal sowing dates")
    plt.xlim((290,400))
    plt.ylim((0,0.5))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates in {typeFarm} farms')
    plt.legend(loc='best')
    

###################### visualize crop calendar and optimal sowing dates against observed
# Fit regression model
# Create the df 
reg_df = obs[['calend', 'sw_doy', 'opt_sw']]
reg_df.dropna(inplace=True) # just for datalines without Nan

# model for crop calendar
model_cal = smf.ols('calend ~ sw_doy', data=reg_df)
res_cal = model_cal.fit()
# Inspect the results
print(res_cal.summary())
pred_ols_cal = res_cal.get_prediction()
iv_l_cal = pred_ols_cal.summary_frame()["obs_ci_lower"]
iv_u_cal = pred_ols_cal.summary_frame()["obs_ci_upper"]

# model for optimal sowing date
model_opt = smf.ols('opt_sw ~ sw_doy', data=reg_df)
res_opt = model_opt.fit()
print(res_opt.summary())
pred_ols_opt = res_opt.get_prediction()
iv_l_opt = pred_ols_opt.summary_frame()["obs_ci_lower"]
iv_u_opt = pred_ols_opt.summary_frame()["obs_ci_upper"]

fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sw_doy'], reg_df['calend'], "x", label="Crop Calendar", color='blue')
plt.plot(reg_df['sw_doy'], reg_df['opt_sw'], "+", label="Optimal Simulated", color='orange')

#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sw_doy'], res_cal.fittedvalues, color='blue', linestyle='-', label="OLS Crop Calendar")
plt.plot(reg_df['sw_doy'], res_opt.fittedvalues, color='orange', linestyle='-', label="OLS Optimal Simulated")
"""
plt.plot(reg_df['sw_doy'], iv_u_cal, color='blue', linestyle='dotted', label='upper/lower conf. interval Cal' )
plt.plot(reg_df['sw_doy'], iv_l_cal, color='blue', linestyle='dotted')
plt.plot(reg_df['sw_doy'], iv_u_opt, color='orange', linestyle='dotted', label='upper/lower conf. interval Opt' )
plt.plot(reg_df['sw_doy'], iv_l_opt, color='orange', linestyle='dotted')
"""
plt.plot([270,500], [270,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((270,500))
plt.ylim((270,500))
plt.xlabel('Observed sowing dates [doy]', {'fontsize': 11})
plt.ylabel('Crop Calendar/Optimal Simulated sowing dates [doy]', {'fontsize': 11})
plt.suptitle('Zimbabwe', fontsize = 13)
plt.title('Observed vs. Crop Calendar resp. Optimal Simulated sowing dates', {'fontsize': 12})
#plt.text(340, 390, f'R2 = {round(res.rsquared,3)}')
#plt.text(340, 380, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Corr_Obs_Zim.png', dpi=260)


############################# visualize sowing dates (observed and simulated and difference) with county boundaries
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp')
gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


fig, ax = plt.subplots(figsize=(10, 8))

plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sw_doy'], vmin=300, vmax=400)
plt.title('Zimbabwe: Observed sowing dates 2021-2022')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['opt_sw'], vmin=300, vmax=400)
#plt.title('Zimbabwe: Optimal sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['calend'], vmin=300, vmax=400)
#plt.title('Zimbabwe: Crop calendar')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['abs_diff'], vmin=0, vmax=60)
#plt.title('Zimbabwe: Absolute difference between optimal and observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['abs_diff_cal'], vmin=0, vmax=60)
#plt.title("Zimbabwe: Absolute difference between crop calendar's and observed sowing dates")

#norm = MidpointNormalize(midpoint=0, vmin=-70, vmax=50)

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff'], norm=norm)
#plt.title('Zimbabwe: Difference between optimal and observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff_cal'], norm=norm)
#plt.title("Zimbabwe: Difference between crop calendar's and observed sowing dates")




plt.colorbar(label='doy')
plt.xlabel('Longitude [°E]')
plt.ylabel('Latitude [°N]')

# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Obs_Zim.png', dpi=260)


plt.show()



#########  same analysis but with the yield values #################################################################################################


####################### regression observed vs simulated
# Fit regression model
# Create the df 
reg_df = obs[['sim_yield_opt_22', 'sim_yield_obs_22', 'sector']]
reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('sim_yield_opt_22 ~ sim_yield_obs_22', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sim_yield_obs_22'], reg_df['sim_yield_opt_22'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sim_yield_obs_22'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sim_yield_obs_22'], iv_u, "r--", label='upper and lower conf. interval')
plt.plot(reg_df['sim_yield_obs_22'], iv_l, "r--")
plt.plot([-0.3,13], [-0.3,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((-0.3,13))
plt.ylim((-0.3,13))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Optimal sowing dates: doy')
plt.title('Regression: optimal vs observed sowing dates')
plt.text(10, 5, f'R2 = {round(res.rsquared,3)}')
plt.text(10, 4, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = reg_df[reg_df['sector'] == typeFarm]
    model = smf.ols('sim_yield_opt_22 ~ sim_yield_obs_22', data=plot_df)
    res = model.fit()
    # Inspect the results
    print(f'{typeFarm}')
    print(res.summary())
    
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    
    plt.plot(plot_df['sim_yield_obs_22'], plot_df['sim_yield_opt_22'], "o", label='data')
    plt.plot(plot_df['sim_yield_obs_22'], res.fittedvalues, "b--.", label="OLS")
    plt.plot(plot_df['sim_yield_obs_22'], iv_u, "r--", label='upper and lower conf. interval')
    plt.plot(plot_df['sim_yield_obs_22'], iv_l, "r--")
    plt.plot([-0.3,13], [-0.3,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
    plt.xlim((-0.3,13))
    plt.ylim((-0.3,13))
    plt.xlabel('Observed sowing dates: doy')
    plt.ylabel('Optimal sowing dates: doy')
    plt.title(f'Regression: optimal vs observed sowing dates, farm type: {typeFarm}')
    plt.text(10, 5, f'R2 = {round(res.rsquared,3)}')
    plt.text(10, 4, f'p value = {round(res.f_pvalue,4)}')
    plt.legend(loc="best")

## one plot of them differentiating communal vs commercial areas
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = reg_df[reg_df['sector'] == 'CA']
    else:
        plot_df = reg_df[reg_df['sector'] != 'CA']
    
    model = smf.ols('sim_yield_opt_22 ~ sim_yield_obs_22', data=plot_df)
    res = model.fit()
    # Inspect the results
    print(f'{typeFarm}')
    print(res.summary())
    
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    
    plt.plot(plot_df['sim_yield_obs_22'], plot_df['sim_yield_opt_22'], "o", label='data')
    plt.plot(plot_df['sim_yield_obs_22'], res.fittedvalues, "b--.", label="OLS")
    plt.plot(plot_df['sim_yield_obs_22'], iv_u, "r--", label='upper and lower conf. interval')
    plt.plot(plot_df['sim_yield_obs_22'], iv_l, "r--")
    plt.plot([-0.3,13], [-0.3,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
    plt.xlim((-0.3,13))
    plt.ylim((-0.3,13))
    plt.xlabel('Observed sowing dates: doy')
    plt.ylabel('Optimal sowing dates: doy')
    plt.title(f'Regression: optimal vs observed sowing dates in {typeFarm} farms')
    plt.text(10,5, f'R2 = {round(res.rsquared,3)}')
    plt.text(10,4, f'p value = {round(res.f_pvalue,4)}')
    plt.legend(loc="best")



######################## regression observed vs crop calendar
# Fit regression model
# Create the df 
reg_df = obs[['sim_yield_cal_22', 'sim_yield_obs_22']]
reg_df.dropna(inplace=True) # just for datalines without Nan

model = smf.ols('sim_yield_cal_22 ~ sim_yield_obs_22', data=reg_df)
res = model.fit()

# Inspect the results
print(res.summary())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sim_yield_obs_22'], reg_df['sim_yield_cal_22'], "o", label="data")
#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sim_yield_obs_22'], res.fittedvalues, "b--.", label="OLS")
plt.plot(reg_df['sim_yield_obs_22'], iv_u, "r--", label='upper and lower conf. interval' )
plt.plot(reg_df['sim_yield_obs_22'], iv_l, "r--")
plt.plot([-0.3,13], [-0.3,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((-0.3,13))
plt.ylim((-0.3,13))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Crop calendar sowing dates: doy')
plt.title('Regression: crop calendar vs observed sowing dates')
plt.text(0, 12, f'R2 = {round(res.rsquared,3)}')
plt.text(0, 11, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")


###################### compare differences diff_calend and diff
comp_diff = obs[['diff_yield_cal', 'diff_yield']]
comp_diff.dropna(inplace=True)
# plt.violinplot([comp_diff['diff_yield_cal'], comp_diff['diff']], showmeans=True, showmedians=True)
plt.boxplot([comp_diff['diff_yield_cal'], comp_diff['diff_yield']], labels=['Crop Calendar', 'Optimal'] )
plt.axhline(0, c='grey', linestyle='dashed')
plt.title("Difference between observed and optimal, resp. crop calendar's sowing dates")
plt.ylabel('days')
#plt.ylim((-90,50))

###################### visualize observed, crop calendar and optimal sowing dates distribution with distribution line
# Create the df 
df_dist = obs[['sim_yield_cal_22', 'sim_yield_obs_22', 'sim_yield_opt_22', 'sector']]
df_dist.dropna(inplace=True)

fig,ax = plt.subplots(figsize=(8, 6))

sns.kdeplot(df_dist['sim_yield_cal_22'], label = "Crop Calendar", fill = True, bw_adjust=2) # bw_adjust ==> to have the line smoother
sns.kdeplot(df_dist['sim_yield_obs_22'], label = "Observation",  fill = True, bw_adjust=2)
sns.kdeplot(df_dist['sim_yield_opt_22'], label = "Optimal sowing dates",  fill = True, bw_adjust=2)
plt.xlim((-2,14))
plt.ylim((0,0.22))
plt.xlabel('Yield: t/ha')
plt.title('Distribution of yields')
plt.legend(loc='best')


## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = df_dist[df_dist['sector'] == typeFarm]
    sns.kdeplot(plot_df['sim_yield_cal_22'], label = "Crop Calendar", fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['sim_yield_obs_22'], label = "Observation",  fill = True, bw_adjust=2)
    sns.kdeplot(plot_df['sim_yield_opt_22'], label = "Optimal sowing dates",  fill = True, bw_adjust=2)
    plt.xlim((-2,14))
    plt.ylim((0,0.22))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, farm type: {typeFarm}')
    plt.legend(loc='best')

## one plot for commercial and one for communal farms
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = df_dist[df_dist['sector'] == 'CA']
    else:
        plot_df = df_dist[df_dist['sector'] != 'CA']
    sns.kdeplot(plot_df['sim_yield_cal_22'], label = "Crop Calendar", fill = True, bw_adjust=1)
    sns.kdeplot(plot_df['sim_yield_obs_22'], label = "Observation",  fill = True, bw_adjust=1)
    sns.kdeplot(plot_df['sim_yield_opt_22'], label = "Optimal sowing dates",  fill = True, bw_adjust=1)    
    plt.xlim((-2,14))
    plt.ylim((0,0.27))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates in {typeFarm} farms')
    plt.legend(loc='best')

## distribution of sowing dates depending on the farmtype and on the datasource
for j in range(0,3):
    datasource = ['sim_yield_cal_22', 'sim_yield_opt_22', 'sim_yield_obs_22'][j]
    fig,ax = plt.subplots(figsize=(8, 6))
    for i in range(0,5):
        typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
        plot_df = df_dist[df_dist['sector'] == typeFarm]
        sns.kdeplot(plot_df[datasource], label = f'{typeFarm}',  fill = True, bw_adjust=2)
    plt.xlim((-2,14))
    plt.ylim((0,0.22))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, {datasource}')
    plt.legend(loc='best')

## distribution of sowing dates depending on the farmtype (communal vs commercial) and on the datasource
for j in range(0,3):
    datasource = ['sim_yield_cal_22', 'sim_yield_opt_22', 'sim_yield_obs_22'][j]
    fig,ax = plt.subplots(figsize=(8, 6))
    for i in range(0,2):
        typeFarm = ['communal', 'commercial'][i]
        if typeFarm == 'communal':
            plot_df = df_dist[df_dist['sector'] == 'CA']
        else:
            plot_df = df_dist[df_dist['sector'] != 'CA']
        sns.kdeplot(plot_df[datasource], label = f'{typeFarm}',  fill = True, bw_adjust=2)
    plt.xlim((-2,14))
    plt.ylim((0,0.22))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, {datasource}')
    plt.legend(loc='best')

###################### visualize observed, crop calendar and optimal sowing dates distribution with histograms (to do better)
# Create the df 
df_dist = obs[['sim_yield_cal_22', 'sim_yield_obs_22', 'sim_yield_opt_22', 'sector']]
df_dist.dropna(inplace=True)

for i in range(0,11):
    fig,ax = plt.subplots(figsize=(8, 6))
    alpha = np.arange(0,1.1,0.1)[i]
    plt.hist(df_dist['sim_yield_obs_22'], bins=33, range=(290,400), histtype='bar', label = "Observation", alpha = alpha, color='blue')
    plt.hist(df_dist['sim_yield_opt_22'], bins=33, range=(290,400), histtype='bar', label = "Optimal sowing dates", alpha = alpha, color='magenta')
    plt.hist(df_dist['sim_yield_cal_22'], bins=33, range=(290,400), histtype='bar', label = "Crop Calendar", alpha = alpha, color='yellow')

    plt.suptitle(f'Alpha = {alpha}')
    plt.xlim((290,400))
    plt.ylim((0,28000))
    plt.xlabel('Sowing date: doy')
    plt.title('Distribution of sowing dates')
    plt.legend(loc='best')


## one plot of them differentiating the type of farm
for i in range(0,5):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['CA', 'SSCFA', 'A2', 'A1', 'LSCFA'][i]
    plot_df = df_dist[df_dist['sector'] == typeFarm]
    sns.distplot(plot_df['sim_yield_cal_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Crop Calendar")
    sns.distplot(plot_df['sim_yield_obs_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Observation")
    sns.distplot(plot_df['sim_yield_opt_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Optimal sowing dates")
    plt.xlim((290,400))
    plt.ylim((0,0.5))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates, farm type: {typeFarm}')
    plt.legend(loc='best')

## one plot for commercial and one for communal farms
for i in range(0,2):
    fig,ax = plt.subplots(figsize=(8, 6))
    typeFarm = ['communal', 'commercial'][i]
    if typeFarm == 'communal':
        plot_df = df_dist[df_dist['sector'] == 'CA']
    else:
        plot_df = df_dist[df_dist['sector'] != 'CA']
        
    sns.distplot(plot_df['sim_yield_cal_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Crop Calendar")
    sns.distplot(plot_df['sim_yield_obs_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Observation")
    sns.distplot(plot_df['sim_yield_opt_22'], hist=False, kde = True, kde_kws = {'linewidth': 1}, label = "Optimal sowing dates")
    plt.xlim((290,400))
    plt.ylim((0,0.5))
    plt.xlabel('Sowing date: doy')
    plt.title(f'Distribution of sowing dates in {typeFarm} farms')
    plt.legend(loc='best')
    

###################### visualize crop calendar and optimal sowing dates against observed
# Fit regression model
# Create the df 
reg_df = obs[['sim_yield_cal_22', 'sim_yield_obs_22', 'sim_yield_opt_22']]
reg_df.dropna(inplace=True) # just for datalines without Nan

# model for crop calendar
model_cal = smf.ols('sim_yield_cal_22 ~ sim_yield_obs_22', data=reg_df)
res_cal = model_cal.fit()
# Inspect the results
print(res_cal.summary())
pred_ols_cal = res_cal.get_prediction()
iv_l_cal = pred_ols_cal.summary_frame()["obs_ci_lower"]
iv_u_cal = pred_ols_cal.summary_frame()["obs_ci_upper"]

# model for optimal sowing date
model_opt = smf.ols('sim_yield_opt_22 ~ sim_yield_obs_22', data=reg_df)
res_opt = model_opt.fit()
print(res_opt.summary())
pred_ols_opt = res_opt.get_prediction()
iv_l_opt = pred_ols_opt.summary_frame()["obs_ci_lower"]
iv_u_opt = pred_ols_opt.summary_frame()["obs_ci_upper"]

fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['sim_yield_obs_22'], reg_df['sim_yield_cal_22'], "x", label="Crop Calendar", color='blue')
plt.plot(reg_df['sim_yield_obs_22'], reg_df['sim_yield_opt_22'], "+", label="Optimal sowing date", color='orange')

#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['sim_yield_obs_22'], res_cal.fittedvalues, color='blue', linestyle='-', label="OLS Crop Calendar")
plt.plot(reg_df['sim_yield_obs_22'], res_opt.fittedvalues, color='orange', linestyle='-', label="OLS Optimal sowing dates")
"""
plt.plot(reg_df['sim_yield_obs_22'], iv_u_cal, color='blue', linestyle='dotted', label='upper/lower conf. interval Cal' )
plt.plot(reg_df['sim_yield_obs_22'], iv_l_cal, color='blue', linestyle='dotted')
plt.plot(reg_df['sim_yield_obs_22'], iv_u_opt, color='orange', linestyle='dotted', label='upper/lower conf. interval Opt' )
plt.plot(reg_df['sim_yield_obs_22'], iv_l_opt, color='orange', linestyle='dotted')
"""
plt.plot([-0.3,13], [-0.3,13], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((-0.3,13))
plt.ylim((-0.3,13))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Crop calendar/optimal sowing dates: doy')
plt.title('Regression: crop calendar resp. optimal vs observed sowing dates')
#plt.text(340, 390, f'R2 = {round(res.rsquared,3)}')
#plt.text(340, 380, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

############################# visualize sowing dates (observed and simulated and difference) with county boundaries
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp')
gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


fig, ax = plt.subplots(figsize=(10, 8))

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sim_yield_obs_22'], vmin=0, vmax=13)
#plt.title('Zimbabwe: Yield from observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sim_yield_opt_22'], vmin=0, vmax=13)
#plt.title('Zimbabwe: Yield from optimal sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sim_yield_cal_22'], vmin=0, vmax=13)
#plt.title('Zimbabwe: Yield from crop calendar')

norm = MidpointNormalize(midpoint=0, vmin=-5, vmax=9)

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff_yield'], norm=norm)
#plt.title('Zimbabwe: Difference between yields from optimal and observed sowing dates')

plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff_yield_cal'], norm=norm)
plt.title("Zimbabwe: Difference between yields from crop calendar's and observed sowing dates")




plt.colorbar(label='t/ha')
plt.xlabel('longitude °E')
plt.ylabel('latitude °N')

# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')

plt.show()