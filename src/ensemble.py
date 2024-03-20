# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:34:17 2023

@author: arulrich

create netcdf files for ensemble of models
"""

# import library
import xarray as xr # v. 0.16.2
import os # 
import numpy as np # v. 1.26.0

def build_dataarray_ensemble(work_dir, sowing_dates):
    """Build a dataset of simulated yield for each sowing date and each season of the ensemble of models.

    Args:
        work_dir (string): Path to the directory containing the NetCDF files
        sowing_dates (List[int]): List of sowing dates

    Returns:
        yield_da: Dataarray (dim: lon, lat, time, and sowing date) containing the simulated yield for each sowing date and each season of the ensemble of models.

    """
    
    # Initialize the dataset
    # Initialize the dataset so that you have the full extent of the area! => find a good method, right now I initialize with a sowing date's dataarray that I know that has the full extent
    yield_ds = xr.Dataset()
    
    for sowing_date in sowing_dates:
        
        # celsius
        outfile_c = os.path.join(work_dir,f'celsius_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_c):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds_c = xr.open_dataset(outfile_c)
        # Remove the last time step from the dataset
        ds_c = ds_c.isel(time=slice(None, -1)) # silence this line if you want simulated yield grown entirely in the last year (e.g. Zimbabwe 2022)
        
        # Read the 'yield' variable
        yield_data_c = ds_c['Yield']
        
        # stics
        outfile_s = os.path.join(work_dir,f'stics_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_s):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds_s = xr.open_dataset(outfile_s)
        # Remove the last time step from the dataset
        ds_s = ds_s.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data_s = ds_s['Yield']
        
        # dssat
        outfile_d = os.path.join(work_dir,f'dssat_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_d):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds_d = xr.open_dataset(outfile_d)
        # Remove the last time step from the dataset
        ds_d = ds_d.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data_d = ds_d['Yield']
        
        # Combine the DataArrays into a single DataArray
        combined_array = xr.concat([yield_data_c, yield_data_s, yield_data_d], dim='concat_dim')
        
        # set all the Nan values to 0 so that crop failure of dssat has value zero instead of nan
        combined_array = combined_array.fillna(0)
        
        # Calculate the mean along the concatenated dimension, ignoring NaN values
        mean_array = np.nanmean(combined_array, axis=0)
        
        # Create a new xarray DataArray with the calculated mean values and the same coordinates
        ensemble_array = xr.DataArray(mean_array, coords={'time': yield_data_c['time'], 'lat': yield_data_c['lat'], 'lon': yield_data_c['lon']}, dims=['time', 'lat', 'lon'])

        # add the array of that sowing date to the dataset
        yield_ds[sowing_date] = ensemble_array
        
        # Close the current dataset
        ds_c.close()
        ds_s.close()
        ds_d.close()
        
    # Merge all the dset variables into one DataArray
    yield_da = yield_ds.to_array(dim='sowing_date')
    
    return yield_da
        

# build array for the single models
def build_dataarray(work_dir, sowing_dates, model):
    """Build a dataarry of simulated yields for each sowing date.

    Args:
        work_dir (string): Path to the directory containing the NetCDF files
        sowing_dates (List[int]): List of sowing dates
        model (string): name of the model (celsius, stics or dssat)

    Returns:
        yield_da: Dataarray (dim: lon, lat, time, and sowing date) containing the simulated yield for each sowing date and each season of a certain model.

    """
    
    # Initialize the datasets
    yield_ds = xr.Dataset()
    
    for sowing_date in sowing_dates:
        
        outfile = os.path.join(work_dir,f'{model}_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds = xr.open_dataset(outfile)
        # Remove the last time step from the dataset
        ds = ds.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data = ds['Yield']
        
        # Add the yield as a data variable to the yield_ds with sowing date as the data variable
        yield_ds[sowing_date] = yield_data
    
        # Close the current dataset
        ds.close()
    
    # Merge all the dset variables into one DataArray
    yield_da = yield_ds.to_array(dim='sowing_date')
        
    return yield_da        

"""
# Zimbabwe
work_dir = "C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/sim_data" 
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# Zimbabwe 2020, 2021, 2022
work_dir = "C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_90_2020_2022/sim_data" 
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_90_2020_2022/output'

# West Africa
work_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/sim_data'
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
work_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/sim_data'
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
work_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/sim_data'
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
"""

sowing_dates = list(range(10, 361, 10))

# ensemble
yield_da = build_dataarray_ensemble(work_dir, sowing_dates)
yield_da.to_netcdf(save_dir+"/ensemble_yield_MgtMais0_allSowingDates_2.0.nc")

# celsius
model = 'celsius'
yield_da = build_dataarray(work_dir, sowing_dates, model)
yield_da.to_netcdf(save_dir+f'/{model}_yield_MgtMais0_allSowingDates_2.0.nc')

# stics
model = 'stics'
yield_da = build_dataarray(work_dir, sowing_dates, model)
yield_da.to_netcdf(save_dir+f'/{model}_yield_MgtMais0_allSowingDates_2.0.nc')

# dssat
model = 'dssat'
yield_da = build_dataarray(work_dir, sowing_dates, model)
yield_da.to_netcdf(save_dir+f'/{model}_yield_MgtMais0_allSowingDates_2.0.nc')
   


def build_dataarray_ensemble_mean_cv(work_dir, sowing_dates):
    """Build a dataarray of mean yield and CV for each sowing date of the ensemble of models.

    Args:
        work_dir (string): Path to the directory containing the NetCDF files
        sowing_dates (List[int]): List of sowing dates

    Returns:
        mean_yield_da, cv_yield_da: Dataarray containing the mean yield and the CV for each sowing date of the ensemble of models.

    """
    
    # Initialize the datasets
    mean_yield_ds = xr.Dataset()
    cv_yield_ds = xr.Dataset()
    
    for sowing_date in sowing_dates:
        
        # celsius
        outfile_c = os.path.join(work_dir,f'celsius_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_c):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds_c = xr.open_dataset(outfile_c)
        # Remove the last time step from the dataset
        ds_c = ds_c.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data_c = ds_c['Yield']
        
        # stics
        outfile_s = os.path.join(work_dir,f'stics_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_s):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds_s = xr.open_dataset(outfile_s)
        # Remove the last time step from the dataset
        ds_s = ds_s.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data_s = ds_s['Yield']
        
        # dssat
        outfile_d = os.path.join(work_dir,f'dssat_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile_d):
            continue
        #    yield_data_d = xr.zeros_like(yield_data_s) # create a dataarray with zeros
        #else:
        # Open the NetCDF dataset for the current sowing date
        ds_d = xr.open_dataset(outfile_d)
        # Remove the last time step from the dataset
        ds_d = ds_d.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data_d = ds_d['Yield']
        
        # Combine the DataArrays into a single DataArray
        combined_array = xr.concat([yield_data_c, yield_data_s, yield_data_d], dim='concat_dim')
        
        # set all the Nan values to 0 so that crop failure of dssat has value zero instead of nan
        combined_array = combined_array.fillna(0)
        
        # Calculate the mean along the concatenated dimension, ignoring NaN values
        mean_array = np.nanmean(combined_array, axis=0)
        
        # Create a new xarray DataArray with the calculated mean values and the same coordinates
        new_array = xr.DataArray(mean_array, coords={'time': yield_data_c['time'], 'lat': yield_data_c['lat'], 'lon': yield_data_c['lon']}, dims=['time', 'lat', 'lon'])
        
        yield_data = new_array

        # Compute the mean yield and the CV along the time dimension
        mean_yield = yield_data.mean(dim='time', skipna=True)
        std_yield = yield_data.std(dim='time', skipna=True)
    
        # Calculate the coefficient of variation (CV)
        cv_yield = std_yield / mean_yield
        
        # Add the mean yield as a data variable to the mean_yield_ds with sowing date as the coordinate
        mean_yield_ds[sowing_date] = mean_yield
        cv_yield_ds[sowing_date] = cv_yield
    
        # Close the current dataset
        ds_c.close()
        ds_s.close()
        ds_d.close()
    
    # Merge all the dset variables into one DataArray
    mean_yield_da = mean_yield_ds.to_array(dim='sowing_date')
    cv_yield_da = cv_yield_ds.to_array(dim='sowing_date')
    
    return mean_yield_da, cv_yield_da




# build array for the single models
def build_dataarray_mean_cv(work_dir, sowing_dates, model):
    """Build a dataarry of mean yield and CV for each sowing date.

    Args:
        work_dir (string): Path to the directory containing the NetCDF files
        sowing_dates (List[int]): List of sowing dates
        model (string): name of the model (celsius, stics or dssat)

    Returns:
        mean_yield_da, cv_yield_da: Dataarray containing the mean yield and the CV for each sowing date

    """
    
    # Initialize the datasets
    mean_yield_ds = xr.Dataset()
    cv_yield_ds = xr.Dataset()
    
    for sowing_date in sowing_dates:
        
        outfile = os.path.join(work_dir,f'{model}_yearly_MgtMais0_{sowing_date}_2.0.nc')
        
        if not os.path.exists(outfile):
            continue
        # Open the NetCDF dataset for the current sowing date
        ds = xr.open_dataset(outfile)
        # Remove the last time step from the dataset
        ds = ds.isel(time=slice(None, -1))
        
        # Read the 'yield' variable
        yield_data = ds['Yield']
        
        # Compute the mean yield and the CV along the time dimension
        mean_yield = yield_data.mean(dim='time', skipna=True)
        std_yield = yield_data.std(dim='time', skipna=True)
    
        # Calculate the coefficient of variation (CV)
        cv_yield = std_yield / mean_yield
        
        # Add the mean yield as a data variable to the mean_yield_ds with sowing date as the coordinate
        mean_yield_ds[sowing_date] = mean_yield
        cv_yield_ds[sowing_date] = cv_yield
    
        # Close the current dataset
        ds.close()
    
    # Merge all the dset variables into one DataArray
    mean_yield_da = mean_yield_ds.to_array(dim='sowing_date')
    cv_yield_da = cv_yield_ds.to_array(dim='sowing_date')
        
    return mean_yield_da, cv_yield_da

# ensemble
my,cv = build_dataarray_ensemble_mean_cv(work_dir, sowing_dates)
my.to_netcdf(save_dir+"/ensemble_yieldMean_MgtMais0_allSowingDates_2.0.nc")
cv.to_netcdf(save_dir+"/ensemble_yieldCV_MgtMais0_allSowingDates_2.0.nc")

# celsius
model = 'celsius'
my,cv = build_dataarray_mean_cv(work_dir, sowing_dates, model)
my.to_netcdf(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
cv.to_netcdf(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')

# stics
model = 'stics'
my,cv = build_dataarray_mean_cv(work_dir, sowing_dates, model)
my.to_netcdf(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
cv.to_netcdf(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')

# dssat
model = 'dssat'
my,cv = build_dataarray_mean_cv(work_dir, sowing_dates, model)
my.to_netcdf(save_dir+f'/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc')
cv.to_netcdf(save_dir+f'/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc')
