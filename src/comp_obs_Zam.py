# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:04:29 2024

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
import numpy as np


"""
# already did it

# load data
obs = pd.read_csv('C:/Users/arulrich/Desktop/MA/survey/RALS_for_YgAnalysis-forArgeo_to use.csv')
obs = obs.iloc[:,1:]

obs.columns
obs.shape
obs.size
obs.head
obs.describe()
obs.info()
duplicate_rows = obs.duplicated()
duplicate_rows.sum()

obs['plant_month'].unique() # 12.,  1., 11., 10.,  2.,  3.,  5.,  4.,  9.,  6., nan,  8.,  7.
obs['plant_week'].unique() # '4th week', '1st week', '3rd week', '2nd week', nan

# add column about coordinates
obs['coord'] = np.nan
for i in range(len(obs)):
    obs['coord'][i] = str(obs['lon'][i]) + '_' + str(obs['lat'][i])  


#### create a column with the sowing dates in doy
obs['sw_doy'] = np.nan

for i in range(len(obs)):
    print(i)
    if obs['plant_month'][i] == 1.: # January
        if obs['plant_week'][i] == '1st week': # 4th of the month
            obs['sw_doy'][i] = 4
        elif obs['plant_week'][i] == '2nd week': # 11th
            obs['sw_doy'][i] = 11
        elif obs['plant_week'][i] == '3rd week': # 18th
            obs['sw_doy'][i] = 18
        elif obs['plant_week'][i] == '4th week': # 25th
            obs['sw_doy'][i] = 25
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 2.: # February
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 35
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 42
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 49
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 56
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 3.: # March
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 63
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 70
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 77
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 84
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 4.: # April
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 94
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 101
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 108
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 115
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 5.: # Mai
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 124
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 131
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 138
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 145
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 6.: # June
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 155
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 162
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 169
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 176
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 7.: # July
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 185
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 192
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 199
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 206
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 8.: # August
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 216
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 223
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 230
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 237
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 9.: # September
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 247
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 254
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 261
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 268
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 10.: # October
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 277
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 284
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 291
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 298
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 11.: # November
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 308
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 315
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 322
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 329
        else:
            obs['sw_doy'][i] = np.nan
    elif obs['plant_month'][i] == 12.: # Dezember
        if obs['plant_week'][i] == '1st week':
            obs['sw_doy'][i] = 338
        elif obs['plant_week'][i] == '2nd week':
            obs['sw_doy'][i] = 345
        elif obs['plant_week'][i] == '3rd week':
            obs['sw_doy'][i] = 352
        elif obs['plant_week'][i] == '4th week':
            obs['sw_doy'][i] = 359
        else:
            obs['sw_doy'][i] = np.nan
    else: 
        obs['sw_doy'][i] = np.nan

obs['sw_doy'].describe()

start_season = 170
def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x

obs['sw_doy'] = obs['sw_doy'].apply(adjust_start)

plt.hist(obs['sw_doy'])#, bins=365, range=(0,365))
plt.title('Distribution of sowing dates')
plt.xlabel('doy')
plt.ylabel('Amount of locations')
plt.ylim((0,50))

# we have farms that have two or more fields pro year and different sowing dates
# here we detect those cases and as sowing date we take the weighted average depending on the field size

# Define a function to calculate the weighted average sowing date and incorporate other observations
def weighted_avg(group):
    # Calculate the total area planted
    total_ha_plant = group['ha_plant'].sum()
    # Calculate the weighted average sowing date using the ha_plant values
    weighted_dates = (group['sw_doy'] * group['ha_plant']).sum() / total_ha_plant
    # Create a dictionary to hold all observations for the group
    group_obs = {}
    # Iterate through other columns and get the observations
    for col in group.columns:
        if col not in ['coord', 'year', 'ha_plant', 'sw_doy']:
            group_obs[col] = group[col].tolist()[0]  # Get the first observation (can be changed based on needs)
    # Return a Pandas Series containing the weighted average sowing date and other observations
    return pd.Series({'sw_doy': weighted_dates, **group_obs})

# Group the DataFrame by coordinate and year, then apply the weighted_avg function to each group
obs = obs.groupby(['coord', 'year']).apply(weighted_avg).reset_index()


###################### create a column with values of optimal sowing dates
obs['opt_sw'] = np.nan


# load data
# optimal sowing dates
model = 'ensemble'
da = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output/{model}_MgtMais0_optSwDateMain_2.0.nc')

df=da.to_dataframe('sowing_date').reset_index()
plt.hist(df['sowing_date'],bins=365)
plt.ylim((0,1))

for i in range(len(obs)):
    obs['opt_sw'][i] = da.sel(lat = obs['lat'][i], lon = obs['lon'][i], method='nearest').values
    

obs['opt_sw'] = obs['opt_sw'].apply(adjust_start)

obs['opt_sw'].describe()
# how many nan values
obs['opt_sw'].isna().sum()


plt.hist(obs['opt_sw'])#, bins=365, range=(0,365))
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
obs['calend'] = np.nan

# load data
# crop calendar
da = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar\\Zam_crop_calendar_0_05.nc')

for i in range(len(obs)):
    obs['calend'][i] = da.sel(lat = obs['lat'][i], lon = obs['lon'][i], method='nearest').values

    
obs['calend'] = obs['calend'].apply(adjust_start)

obs['calend'].describe()
# how many nan values
obs['calend'].isna().sum()


plt.hist(obs['calend'])#, bins=38, range=(240,620))
plt.title('Distribution of sowing dates in Crop Calendar')
plt.xlabel('doy')
plt.ylabel('Amount of locations')
plt.xlim((240,620))


######################### create a column with absolute difference between obs and crop calendar sowing date
obs['abs_diff_cal'] = ((obs['calend'] - obs['sw_doy'])**2)**(1/2)

obs['abs_diff_cal'].describe()


plt.hist(obs['abs_diff_cal'], bins=9, histtype='stepfilled')
plt.title('Distribution of absolute difference between sowing dates: crop calendar - observed')
plt.xlabel('doy')
plt.ylabel('Amount of locations')


######################### create a column with absolute difference between obs and crop calendar sowing date
obs['diff_cal'] = obs['calend'] - obs['sw_doy']

obs['diff_cal'].describe()


plt.hist(obs['diff_cal'], bins=37)
plt.title('Distribution of difference between sowing dates: crop calendar - observed')
plt.xlabel('doy')
plt.ylabel('Amount of locations')

############################# add simulated yield given from the sowing date
def extract_yearly_yield(dataarray, year, sowing_date, lat, lon):
    year_data = dataarray.sel(time=year)

    # Interpolate with linear method along 'sowing_date'
    interpolated_data = year_data.interp(sowing_date=sowing_date, method='linear')
    
    # Interpolate with nearest method along 'lat' and 'lon'
    value_at_location = interpolated_data.interp(lat=lat, lon=lon, method='nearest')
    
    return value_at_location

# load dataarray with the simulated yield values for each sowing date, year, and location (we need sowing date of the growing season 2021/22)
model = 'ensemble'
dataarray = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output/{model}_yield_MgtMais0_allSowingDates_2.0.nc')
dataarray = dataarray.where(dataarray != 0) # set 0 values back to Nan

# add column about the year of the sowing date
obs['plant_year_obs'] = np.nan
obs['plant_year_opt'] = np.nan
obs['plant_year_cal'] = np.nan


for i in range(len(obs)):
    if obs['sw_doy'][i] > 365:
        obs['plant_year_obs'][i] = obs['year'][i]
    elif obs['sw_doy'][i] <= 365:
        obs['plant_year_obs'][i] = obs['year'][i]-1
        
for i in range(len(obs)):
    if obs['opt_sw'][i] > 365:
        obs['plant_year_opt'][i] = obs['year'][i]
    elif obs['opt_sw'][i] <= 365:
        obs['plant_year_opt'][i] = obs['year'][i]-1
        
for i in range(len(obs)):
    if obs['calend'][i] > 365:
        obs['plant_year_cal'][i] = obs['year'][i]
    elif obs['calend'][i] <= 365:
        obs['plant_year_cal'][i] = obs['year'][i]-1

obs['sim_yield_obs'] = np.nan
obs['sim_yield_opt'] = np.nan
obs['sim_yield_cal'] = np.nan

# for survey observation 
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['sw_doy'][i]):
        obs['sim_yield_obs'][i] = np.nan
    else:
        if np.isnan(obs['opt_sw'][i]):
            obs['sim_yield_obs'][i] = np.nan
        else:
            obs['sim_yield_obs'][i] = extract_yearly_yield(dataarray=dataarray, year=obs['plant_year_obs'][i], sowing_date = obs['sw_doy'][i], lat= obs['lat'][i], lon= obs['lon'][i])

plt.hist(obs['sim_yield_obs'], bins=200)
plt.xlim((0,14))
plt.ylim((0,800))
obs['sim_yield_obs'].describe()

# for optimal sowing dates
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['opt_sw'][i]):
        obs['sim_yield_opt'][i] = np.nan
    else:
        obs['sim_yield_opt'][i] = extract_yearly_yield(dataarray=dataarray, year=obs['plant_year_opt'][i], sowing_date = obs['opt_sw'][i], lat= obs['lat'][i], lon= obs['lon'][i])

plt.hist(obs['sim_yield_opt'], bins=200)
plt.xlim((0,14))
plt.ylim((0,800))
obs['sim_yield_opt'].describe()

# for crop calendar sowing dates
for i in range(len(obs)):
    print(f'{i} of {len(obs)}')
    if np.isnan(obs['calend'][i]):
        obs['sim_yield_cal'][i] = np.nan
    else:
        if np.isnan(obs['opt_sw'][i]):
            obs['sim_yield_cal'][i] = np.nan
        else:
            obs['sim_yield_cal'][i] = extract_yearly_yield(dataarray=dataarray, year=obs['plant_year_cal'][i], sowing_date = obs['calend'][i], lat= obs['lat'][i], lon= obs['lon'][i])


plt.hist(obs['sim_yield_cal'], bins=200)
plt.xlim((0,14))
plt.ylim((0,800))
obs['sim_yield_cal'].describe()



######################## save resp. load the dataframe containing the data of the survey and from our interpolations
obs.to_csv('C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/planting_dates_zam_and_our_data.csv')
"""

# load it
obs = pd.read_csv('C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/planting_dates_zam_and_our_data.csv')
obs = obs.iloc[:,1:]

####################### regression observed vs simulated
# Fit regression model
# chose the year
year = 2012 # 2012, 2015 or 2019
# Fit regression model
# Create the df 
reg_df = obs[obs['year']==year]
reg_df = reg_df[['opt_sw', 'sw_doy']]
reg_df.dropna(inplace=True) # just for datalines without Nan

"""
# Create the df 
reg_df = obs[['opt_sw', 'sw_doy']]
reg_df.dropna(inplace=True) # just for datalines without Nan
"""

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
plt.plot([180,550], [180,550], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((180,550))
plt.ylim((180,550))
plt.xlabel('Observed sowing dates: doy')
plt.ylabel('Optimal sowing dates: doy')
plt.title('Regression: optimal vs observed sowing dates')
plt.text(460, 380, f'R2 = {round(res.rsquared,3)}')
plt.text(460, 360, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

####################### visualize where are the location and our crop mask
model = 'ensemble'
da = xr.open_dataarray(f'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output/{model}_MgtMais0_optSwDateMain_2.0.nc')


dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp')
gdf11 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')

mask = da.fillna(0)
mask = mask.where(mask == 0, other=1) # give the value 1 to the pixels which not fullfill the condition == 0
mask = mask.where(mask != 0) # give the value nan to the pixels which not fullfill the condition != 0
cmap = mcolors.ListedColormap(['white', 'lightgrey'])

fig, ax = plt.subplots(figsize=(10, 8))

plt.plot(obs['lon'], obs['lat'], marker = '+', c = 'red', linestyle = '', markersize = 1, label = 'Observed farms') # all the farms

mask.plot(cmap=cmap, label='Crop mask')
plt.title('Zambia: Observed sowing dates and crop mask', {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')
plt.legend(loc='best', markerscale = 11)

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Obs_crop_mask_Zam.png', dpi=260)


######################## regression observed vs crop calendar
# chose the year
year = 2012 # 2012, 2015 or 2019
# Fit regression model
# Create the df 
reg_df = obs[obs['year']==year]
reg_df = reg_df[['calend', 'sw_doy']]
reg_df.dropna(inplace=True) # just for datalines without Nan

"""
reg_df = obs[['calend', 'sw_doy']]
reg_df.dropna(inplace=True) # just for datalines without Nan
"""

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
plt.plot([180,550], [180,550], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((180,550))
plt.ylim((180,550))
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

# Fit regression model
# Create the df 
df_dist = obs[['calend', 'sw_doy', 'opt_sw', 'year']]
df_dist.dropna(inplace=True)
df_dist_12 = df_dist[obs['year']==2012]
df_dist_15 = df_dist[obs['year']==2015]
df_dist_19 = df_dist[obs['year']==2019]

df_dist_19['sw_doy'].describe()
df_dist_12['opt_sw'].describe()
df_dist_12['calend'].describe()

fig,ax = plt.subplots(figsize=(8, 6))

sns.kdeplot(df_dist['calend'], label = "Crop Calendar", fill = True, bw_adjust=1.5) # bw_adjust ==> to have the line smoother
sns.kdeplot(df_dist['opt_sw'], label = "Optimal Simulated",  fill = True, bw_adjust=1.5)
sns.kdeplot(df_dist['sw_doy'], label = "Observation",  fill = True, bw_adjust=1.5)
"""
sns.kdeplot(df_dist_12['sw_doy'], label = "Observation 2012",  fill = True, bw_adjust=1.5)
sns.kdeplot(df_dist_15['sw_doy'], label = "Observation 2015",  fill = True, bw_adjust=1.5)
sns.kdeplot(df_dist_19['sw_doy'], label = "Observation 2019",  fill = True, bw_adjust=1.5)
"""
plt.xlim((200,500))
#plt.ylim((0,0.12))
plt.xlabel('Sowing date [doy]', {'fontsize': 12})
plt.ylabel('Density', {'fontsize': 12})
plt.title('Zambia: Distribution of sowing dates', {'fontsize': 15})
plt.legend(loc='best')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Distrib_Obs_Zam.png', dpi=260)



###################### visualize observed, crop calendar and optimal sowing dates distribution with histograms (to do better)
# Create the df 
df_dist = obs[['calend', 'sw_doy', 'opt_sw', 'year']]
df_dist.dropna(inplace=True)

df_dist['sw_doy'].describe()
df_dist['opt_sw'].describe()
df_dist['calend'].describe()

df_dist_12 = df_dist[obs['year']==2012]
df_dist_15 = df_dist[obs['year']==2015]
df_dist_19 = df_dist[obs['year']==2019]

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
    

###################### regression : visualize crop calendar and optimal sowing dates against observed
# Fit regression model
# chose the year
year = 2019 # 2012, 2015 or 2019

# Create the df 
reg_df = obs[obs['year']==year]
reg_df = reg_df[['calend', 'sw_doy', 'opt_sw']]
reg_df.dropna(inplace=True) # just for datalines without Nan

"""
# Create the df 
reg_df = obs[['calend', 'sw_doy', 'opt_sw']]
reg_df.dropna(inplace=True) # just for datalines without Nan
"""

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
plt.plot([180,550], [180,550], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((180,550))
plt.ylim((180,550))
plt.xlabel('Observed sowing dates [doy]', {'fontsize': 11})
plt.ylabel('Crop Calendar/Optimal Simulated sowing dates [doy]', {'fontsize': 11})
plt.suptitle('Zambia', fontsize = 13)
plt.title('Observed vs. Crop Calendar resp. Optimal Simulated sowing dates', {'fontsize': 12})
#plt.text(340, 390, f'R2 = {round(res.rsquared,3)}')
#plt.text(340, 380, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Corr_Obs_Zam.png', dpi=260)


############################# visualize sowing dates (observed and simulated and difference) with county boundaries
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'
gdf01 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp')
gdf11 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

fig, ax = plt.subplots(figsize=(10, 8))

plt.scatter(obs['lon'], obs['lat'], s=1, cmap='plasma', marker = 'o', c = obs['sw_doy'], vmin=260, vmax=410)
plt.title('Zambia: Observed sowing dates 2011/2012, 2014/2015, and 2018/2019')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['opt_sw'], vmin=300, vmax=400)
#plt.title('Zambia: Optimal sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['calend'], vmin=300, vmax=400)
#plt.title('Zambia: Crop calendar')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['abs_diff'], vmin=0, vmax=60)
#plt.title('Zambia: Absolute difference between optimal and observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['abs_diff_cal'], vmin=0, vmax=60)
#plt.title("Zambia: Absolute difference between crop calendar's and observed sowing dates")

#norm = MidpointNormalize(midpoint=0, vmin=-100, vmax=100)

#plt.scatter(obs['lon'], obs['lat'], s=1, cmap='coolwarm', marker = 'o', c = obs['diff'], norm=norm)
#plt.title('Zambia: Difference between optimal and observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=1, cmap='coolwarm', marker = 'o', c = obs['diff_cal'], norm=norm)
#plt.title("Zambia: Difference between crop calendar's and observed sowing dates")

plt.colorbar(label='doy')
plt.xlabel('Longitude [°E]')
plt.ylabel('Latitude [°N]')

# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/Obs_Zam.png', dpi=260)

plt.show()


###################################################################################
#########  same analysis but with the yield values #################################################################################################
###################################################################################

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
#plt.title('Zambia: Yield from observed sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sim_yield_opt_22'], vmin=0, vmax=13)
#plt.title('Zambia: Yield from optimal sowing dates')

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='Spectral', marker = 'o', c = obs['sim_yield_cal_22'], vmin=0, vmax=13)
#plt.title('Zambia: Yield from crop calendar')

norm = MidpointNormalize(midpoint=0, vmin=-5, vmax=9)

#plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff_yield'], norm=norm)
#plt.title('Zambia: Difference between yields from optimal and observed sowing dates')

plt.scatter(obs['lon'], obs['lat'], s=0.5, cmap='coolwarm', marker = 'o', c = obs['diff_yield_cal'], norm=norm)
plt.title("Zambia: Difference between yields from crop calendar's and observed sowing dates")




plt.colorbar(label='t/ha')
plt.xlabel('longitude °E')
plt.ylabel('latitude °N')

# add shp
gdf11.boundary.plot(ax=ax, linewidth=1, color='blue')
gdf01.boundary.plot(ax=ax, linewidth=2, color='black')

plt.show()