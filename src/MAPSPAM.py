# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:36:47 2024

@author: arulrich
"""

# MAPSTAT
# import tif file of maize physical area of rainfed agriculture and convert it to netcdf
# conversion .tif to .nc
from osgeo import gdal
import os 
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import rioxarray

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'

""" # already did it
def tif_to_nc(tif_file_path):
    outputfile = os.path.splitext(tif_file_path)[0] + '.nc'
    #Do not change this line, the following command will convert the geoTIFF to a netCDF
    gdal.Translate(outputfile, tif_file_path, format='NetCDF')

tif_to_nc('C:/Users/arulrich/Documents/data_analysis/MAPSPAM/spam2017V2r1_SSA_A_MAIZ_R.tif')


ds = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/MAPSPAM/spam2017V2r1_SSA_A_MAIZ_R.nc')

da = ds['Band1']

"""

""" already did it for all the areas

# upload the yields from optimal sowing dates
opt_y = xr.open_dataset(save_dir + '/ensemble_yieldMean_MgtMais0_optSowingDates_2.0.nc')
cal_y = xr.open_dataset(save_dir + '/ensemble_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

opt = opt_y['Yield_average']
cal = cal_y['Yield_average']

# new coordinate for both array with extent of opt_y/cal and resolution of da
new_lat = np.arange(opt['lat'][0], opt['lat'][len(opt['lat'])-1], da['lat'][1] - da['lat'][0]) 
new_lon = np.arange(opt['lon'][0], opt['lon'][len(opt['lon'])-1], da['lon'][1] - da['lon'][0])

new_da = da.interp(lat=new_lat, lon=new_lon, method="linear")
new_opt = opt.interp(lat=new_lat, lon=new_lon, method = 'linear')
new_cal = cal.interp(lat=new_lat, lon=new_lon, method = 'linear')


new_da.plot()
new_opt.plot()
new_cal.plot()

production_opt = new_da*new_opt
production_opt.plot()

production_cal = new_da*new_cal
production_cal.plot()

# mask the pixel that are nan in production_opt
nan_mask= np.isnan(production_opt)
production_cal = production_cal.where(~nan_mask)

# save the production netcdf of maize t
production_opt.to_netcdf(save_dir+'/ensemble_productionMean_MgtMais0_optSowingDates_2.0.nc')
production_cal.to_netcdf(save_dir+'/ensemble_productionMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

"""
# load production dataset
production_opt = xr.open_dataset(save_dir+'/ensemble_productionMean_MgtMais0_optSowingDates_2.0.nc')
production_cal = xr.open_dataset(save_dir+'/ensemble_productionMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc')

# load political boundaries
# load data shp
dir_shp = 'C:/Users/arulrich/Documents/data_analysis/shapefiles/'

region = "Rwanda & Burundi"


if region == 'Zimbabwe':
    gdf01 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp') # Zimbabwe
    gdf02 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_0.shp') # Mozambique
    gdf03 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp') # Zambia
    gdf04 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_0.shp') # South Afrika
    gdf05 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_0.shp') # Botswana
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_ZAF_shp/gadm41_ZAF_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15], ignore_index=True)
    
elif region == 'Zambia':
    gdf01 = gpd.read_file(dir_shp + 'gadm41_ZMB_shp/gadm41_ZMB_0.shp') # Zambia
    gdf02 = gpd.read_file(dir_shp + 'gadm41_ZWE_shp/gadm41_ZWE_0.shp') # Zimbabwe
    gdf03 = gpd.read_file(dir_shp + 'gadm41_MOZ_shp/gadm41_MOZ_0.shp') # Mozambique
    gdf04 = gpd.read_file(dir_shp + 'gadm41_BWA_shp/gadm41_BWA_0.shp') # Botswana
    gdf05 = gpd.read_file(dir_shp + 'gadm41_MWI_shp/gadm41_MWI_0.shp') # Malawi
    gdf06 = gpd.read_file(dir_shp + 'gadm41_NAM_shp/gadm41_NAM_0.shp') # Namibia
    gdf07 = gpd.read_file(dir_shp + 'gadm41_AGO_shp/gadm41_AGO_0.shp') # Angola
    gdf08 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_0.shp') # DR Congo
    gdf09 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_0.shp') # Tanzania
    
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
    
elif region == 'West Africa':
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
    gdf11 = pd.concat([gdf14,gdf110], ignore_index=True) # just Togo and Ghana togheter
    
elif region == 'East Africa':
    gdf01 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_0.shp') # DR Congo
    gdf02 = gpd.read_file(dir_shp + 'gadm41_BDI_shp/gadm41_BDI_0.shp') # Burundi
    gdf03 = gpd.read_file(dir_shp + 'gadm41_KEN_shp/gadm41_KEN_0.shp') # Kenia
    gdf04 = gpd.read_file(dir_shp + 'gadm41_RWA_shp/gadm41_RWA_0.shp') # Rwanda
    gdf05 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_0.shp') # Tanzania
    gdf06 = gpd.read_file(dir_shp + 'gadm41_UGA_shp/gadm41_UGA_0.shp') # Uganda
    
    gdf11 = gpd.read_file(dir_shp + 'gadm41_COD_shp/gadm41_COD_1.shp')
    gdf12 = gpd.read_file(dir_shp + 'gadm41_BDI_shp/gadm41_BDI_1.shp')
    gdf13 = gpd.read_file(dir_shp + 'gadm41_KEN_shp/gadm41_KEN_1.shp')
    gdf14 = gpd.read_file(dir_shp + 'gadm41_RWA_shp/gadm41_RWA_1.shp')
    gdf15 = gpd.read_file(dir_shp + 'gadm41_TZA_shp/gadm41_TZA_1.shp')
    gdf16 = gpd.read_file(dir_shp + 'gadm41_UGA_shp/gadm41_UGA_1.shp')
    
    gdf_all = pd.concat([gdf11, gdf12, gdf13, gdf14, gdf15, gdf16], ignore_index=True)
    gdf11 = pd.concat([gdf12,gdf14], ignore_index=True) # just Burundi and Rwanda togheter



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

def gdf_add_production(dataset_opt, dataset_cal, country_shape):
    """Add to shapefile the maize production for each county of the countries within the area

    Args:
        dataset_opt (xr.dataset): dataset with simulated yields from optimal sowing dates
        dataset_opt (xr.dataset): dataset with simulated yields from crop calendar sowing dates
        shape (geopandas shapefile): gpd object with polygon or multipolygon defining the area of counties of a country

    Returns:
        clip_dset: a dataset where there are data only within the shape's area, elsewhere is nan
    """
    reg = country_shape['NAME_1'].values

    # add a column to gdf with the value of internal production
    country_shape['Production_opt'] = np.nan
    
    for i in range(len(country_shape)):
        da = clip_dset(dataset_opt, country_shape[country_shape['NAME_1'] == reg[i]])['__xarray_dataarray_variable__']
        country_shape.loc[country_shape['NAME_1'] == reg[i],'Production_opt'] = da.sum().values
    
    # add a column to gdf with the value of internal production
    country_shape['Production_cal'] = np.nan
    
    for i in range(len(country_shape)):
        da = clip_dset(dataset_cal, country_shape[country_shape['NAME_1'] == reg[i]])['__xarray_dataarray_variable__']
        country_shape.loc[country_shape['NAME_1'] == reg[i],'Production_cal'] = da.sum().values
    
    # add column about the differences
    country_shape['Production_diff'] = country_shape['Production_opt'] - country_shape['Production_cal']
    
    # add a column of relative maize differenz
    country_shape['Production_rel_diff'] = country_shape['Production_diff']/country_shape['Production_opt']

    return country_shape

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

gdf11 = gdf_add_production(production_opt, production_cal, gdf11)

# visualize them
x_min = production_opt['lon'][0]
x_max = production_opt['lon'][-1]
y_min = production_opt['lat'][0]
y_max = production_opt['lat'][-1]

# 'Maize production simulated with optimal sowing dates in Mt'
gdf11['Production_opt'].describe()
norm = Normalize(vmin=0, vmax=260)
fig, ax = plt.subplots()
gdf11.plot(ax=ax, column=gdf11['Production_opt']/1000, cmap='viridis',legend = True, norm = norm)
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
gdf04.boundary.plot(ax=ax, linewidth=1, color='black')
gdf02.boundary.plot(ax=ax, linewidth=1, color='black')

#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))
plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/prod_opt_EA.png', dpi=260)

plt.show()

# 'Maize production simulated with crop calendar sowing dates in Mt'
gdf11['Production_cal'].describe()
fig, ax = plt.subplots()
gdf11.plot(ax=ax, column=gdf11['Production_cal']/1000, cmap='viridis',legend = True, norm = norm)
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
gdf04.boundary.plot(ax=ax, linewidth=1, color='black')
gdf02.boundary.plot(ax=ax, linewidth=1, color='black')
#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/prod_cal_EA.png', dpi=260)

plt.show()

# Difference between maize production simulated with optimal simulated and crop calendar sowing dates in Mt'
gdf11['Production_diff'].describe()
norm = MidpointNormalize(midpoint=0, vmin=0, vmax=60)
fig, ax = plt.subplots()
gdf11.plot(ax=ax, column=gdf11['Production_diff']/1000, cmap='coolwarm',legend = True, norm = norm)
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})
gdf04.boundary.plot(ax=ax, linewidth=1, color='black')
gdf02.boundary.plot(ax=ax, linewidth=1, color='black')
#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))



plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/prod_abs_diff_opt_cal_EA.png', dpi=260)

plt.show()

# Relative difference between maize production simulated with optimal simulated and crop calendar sowing dates in %'
gdf11['Production_rel_diff'].describe()

norm = MidpointNormalize(midpoint=0, vmin=0, vmax=60)
fig, ax = plt.subplots()
gdf11.plot(ax=ax, column=gdf11['Production_rel_diff']*100, cmap='YlOrRd',legend = True, vmin=0, vmax=58)
gdf02.boundary.plot(ax=ax, linewidth=1, color='black')
gdf04.boundary.plot(ax=ax, linewidth=1, color='black')

#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/prod_rel_diff_opt_cal_EA.png', dpi=260)

plt.show()
