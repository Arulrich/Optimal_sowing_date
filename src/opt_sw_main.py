# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:49:36 2023

@author: arulrich

define optimal sowing date for the main season
"""

import pandas as pd # v. 2.1.0
import xarray as xr #  v. 0.16.2

# the difference and the average between sowing dates is calculate differently as usual so that 
# also sowing dates from the end of a year and from the start of the following year can be calculated properly 
def diff_sw_date(swa,swb):
    """Determines the difference between two sowing dates.

    Args:
        swa,swb (int): sowing dates in doy

    Returns:
        diff: difference between the two sowing dates in days
    """
    sowing = [swa,swb]
    diff = min([max(sowing)-min(sowing), 365+min(sowing)-max(sowing)])
    return diff

def mean_sw_date(swa,swb):
    """Determines the mean between two sowing dates.

    Args:
        swa,swb (int): sowing dates in doy

    Returns:
        aver: mean value between the two sowing dates in days
    """
    diff = ((swa - swb)**2)**(1/2)
    if diff < 182:
        aver = (swa + swb)/2
    else:
        aver = (swa + swb + 365)/2
    if aver > 365:
        aver-=365
    return int(round(aver,0))

def def_opt_sw(overlap):
    """Determines the optimal sowing date

    Args:
        overlap (dataframe): overlap of the two sowing windows from yield mean and cv

    Returns:
        opt_sw (int): optimal sowing day in doy
    """
    if len(overlap) == 1:
        opt_sw = overlap.iloc[0,0]
    elif len(overlap) >= 2:
        opt_sw = mean_sw_date(overlap.iloc[0,0], overlap.iloc[1,0])
    return opt_sw


def sw_date_optimization(da_y, da_cv, maturation_time, ranking):
    """Determines for each pixel if it is possible do two cropping season and it define one or resp. two optimal sowing dates.

    Args:
        da_y (xr.Dataarray): Dataarray containing the mean yield for each sowing date
        da_cv (xr.Dataarray): Dataarray containing the cv yield for each sowing date
        maturation_time (int): The minimum time required between planting and harvest for the specific crop
        ranking (int): The minimum number of sowing dates to consider in a sowing window

    Returns:
        results_df_1 (dataframe): Dataframe containing lon, lat, sowing dates for pixel with one cropping season.
        results_df_2 (dataframe): Dataframe containing lon, lat, sowing dates (1 and 2) for pixel with two cropping season.
    """
   
    # Initialize empty DataFrames with column names
    columns1 = ['lat', 'lon', 'sowing_date']
    result_df_1 = pd.DataFrame(columns=columns1)
    columns2 = ['lat', 'lon', 'sowing_date_1', 'sowing_date_2']
    result_df_2 = pd.DataFrame(columns=columns2)

    # pixelwise
    for lat in range(len(da_y['lat'])):
        for lon in range(len(da_y['lon'])):
            print('lat: '+str(lat)+' lon: '+str(lon))
            rank = ranking-1
            # yield
            da_y_sel = da_y.isel(lat=lat, lon=lon) # select pixel values
            if da_y_sel.isnull().all(): 
                result_df_1.append({'lat': da_y_sel['lat'].values, 'lon': da_y_sel['lon'].values, 'sowing_date': float('nan')}, ignore_index=True)
            else:   
                df_y = da_y_sel.to_dataframe(name='Yield') # change format to df
                df_y = df_y.reset_index()
                # two cropping season possible? 
                #double = double_season(df_y, maturation_time) function to write maybe based on a peak investigator
                double = False
                if double == True:
                    print("double")
                    # optimization for two cropping seasons
                else: # optimization for one cropping season
                    df_y_sort = df_y.sort_values(by='Yield', ascending=False) # sort descending
                    # cv
                    da_cv_sel = da_cv.isel(lat=lat, lon=lon)
                    df_cv = da_cv_sel.to_dataframe(name='CV')
                    df_cv = df_cv.reset_index()
                    df_cv_sort = df_cv.sort_values(by='CV', ascending=True) # sort ascending
                    
                    
                    # A) solution if the overlapping sowing wondow is empty: use the sowing window of high yields
                    rank+=1
                    # sowing window Yield
                    sw_range_y = df_y_sort.iloc[:rank] # select first 'rank' rows
                    # sowing window CV
                    sw_range_cv = df_cv_sort.iloc[:rank]
                    overlap = pd.merge(sw_range_y, sw_range_cv, on='sowing_date', how='inner')
                    if overlap.empty:
                        overlap = sw_range_y
                    
                    """ # B) solution if the overlapping sowing wondow is empty: make wider sowing windows
                    # initialise empty overlap df
                    overlap = pd.DataFrame()
                    while overlap.empty:
                        rank+=1
                        # sowing window Yield
                        sw_range_y = df_y_sort.iloc[:rank] # select first 'rank' rows
                        # sowing window CV
                        sw_range_cv = df_cv_sort.iloc[:rank] 
                        # overlap
                        overlap = pd.merge(sw_range_y, sw_range_cv, on='sowing_date', how='inner') # overlap of sowing windows
                    """
                    # decide optimal sowing date
                    opt_sw = def_opt_sw(overlap)
                    
                    # get index for the selection below
                    first_row_name = overlap.index[0]
                    lat_column = [col for col in overlap.columns if 'lat' in col][0]    
                    lon_column = [col for col in overlap.columns if 'lon' in col][0]
                    
                    result_df_1 = result_df_1.append({'lat': overlap.loc[first_row_name,lat_column], 'lon': overlap.loc[first_row_name,lon_column], 'sowing_date': opt_sw}, ignore_index=True)
        
    return result_df_1, result_df_2

# get a dataframe with lon, lat, optimal sowing date and if double cropping is possible

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

model = 'ensemble' # celsius, stics or dssat

da_y = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
da_y = da_y.where(da_y != 0) # set 0 values back to Nan
da_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
da_cv = da_cv.where(da_cv != 0)

# run calculation
opt_sw = sw_date_optimization(da_y, da_cv, maturation_time = 105, ranking=6)[0]

# save
opt_sw.to_csv(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.csv')

for i in range(0,3):
    model = ['celsius', 'stics', 'dssat'][i]
    print(f'########################################  {model} ######################')
    
    da_y = xr.open_dataarray(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
    da_cv = xr.open_dataarray(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
    
    opt_sw = sw_date_optimization(da_y, da_cv, maturation_time = 105, ranking=6)[0]
    
    opt_sw.to_csv(save_dir+f'/{model}_MgtMais0_optSwDateMain_2.0.csv')


