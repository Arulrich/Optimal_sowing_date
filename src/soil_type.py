# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:25:01 2024

@author: arulrich
"""

# soil type

# import tif file of maize physical area of rainfed agriculture and convert it to netcdf
# conversion .tif to .nc
from osgeo import gdal
import os 
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.colors as mcolors
import geopandas as gpd




# transform the tif file to the right coordinates (no Lambert!)

def trans_crs(tif_file_path):
    input_file = tif_file_path
    output_file = os.path.splitext(tif_file_path)[0] + '_lon_lat.tif'

    # Define source and target CRS
    source_crs = 'EPSG:4326'  # Lambert Azimuthal Equal-Area (LAEA)
    target_crs = 'EPSG:4326'
    
    # Open the input file
    with rasterio.open(input_file) as src:
        # Get the metadata for the input file
        profile = src.profile
        
        # Reproject the dataset
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )


# convert a tif file to a netcdf
def tif_to_nc(tif_file_path):
    outputfile = os.path.splitext(tif_file_path)[0] + '.nc'
    #Do not change this line, the following command will convert the geoTIFF to a netCDF
    gdal.Translate(outputfile, tif_file_path, format='NetCDF')

""" already did it
# Root zone total plant available water holding capacity aggregated at ERZD
input_file = 'C:/Users/arulrich/Documents/data_analysis/gyga_af_agg_erzd_tawcpf23mm__m_1km.tif'
trans_crs(input_file)

tif_to_nc('C:/Users/arulrich/Documents/data_analysis/gyga_af_agg_erzd_tawcpf23mm__m_1km.tif')
tif_to_nc('C:/Users/arulrich/Documents/data_analysis/gyga_af_agg_erzd_tawcpf23mm__m_1km_lon_lat.tif')
"""

# load the transformed netcdf file
ds_trans = xr.open_dataset('C:/Users/arulrich/Documents/data_analysis/gyga_af_agg_erzd_tawcpf23mm__m_1km_lon_lat.nc')

da_trans = ds_trans['Band1']
#da_trans.plot()

# decrease the resolution from factor 5
coarsened_data = da_trans.coarsen(lat=5, lon=5, boundary="trim").mean()


# load dataarray with sowing dates
region = 'Zambia'

# Zimbabwe
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/output'
cal_sw = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zim_crop_calendar_0_05.nc')

# West Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/WA_Celsius_Dssat_Stics_outputs_90/output'
cal_sw = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/WA_crop_calendar_0_05.nc')

# Zambia
save_dir = 'C:/Users/arulrich/Documents/data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/output'
cal_sw = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/Zam_crop_calendar_0_05.nc')

# East Africa
save_dir = 'C:/Users/arulrich/Documents/data_analysis/EA_Celsius_Dssat_Stics_outputs_90/output'
cal_sw = xr.open_dataarray('C:/Users/arulrich/Documents/data_analysis/Crop_calendar/EA_crop_calendar_0_05.nc')


opt_sw = xr.open_dataarray(save_dir + '/ensemble_MgtMais0_optSwDateMain_2.0.nc')

""" try at smaller scale WA 8 < lat < 10°N, where there is only one AEZ (Tropic_warm_subhumid)
opt_sw = opt_sw.sel(lat=slice(8,10))
cal_sw = cal_sw.sel(lat=slice(8,10))
"""


# set coordinates and extent based on opt_sw to reformat the Root zone total plant available water holding capacity array
new_lat = np.arange(opt_sw['lat'][0], opt_sw['lat'][-1] + (opt_sw['lat'][1] - opt_sw['lat'][0]), opt_sw['lat'][1] - opt_sw['lat'][0]) 
new_lon = np.arange(opt_sw['lon'][0], opt_sw['lon'][-1], opt_sw['lon'][1] - opt_sw['lon'][0])

wfc_da = coarsened_data.interp(lat=new_lat, lon=new_lon, method="linear")


####### regression 
# adjust start of the season for the entire area
start_season = 250 # 250 for Zimbabwe, 170 for Zambia, 0 for EA and WA

def adjust_start(x):
    adjusted_x = np.where(x <= start_season, x + 365, x)
    return adjusted_x

opt_sw = xr.apply_ufunc(adjust_start, opt_sw)
cal_sw = xr.apply_ufunc(adjust_start, cal_sw)

# extract variables' values
opt_sw_df = opt_sw.to_dataframe(name='sowing_date').reset_index()
df_cal  = cal_sw.to_dataframe(name='sowing_date_cal').reset_index()
wfc_df = wfc_da.to_dataframe(name='plant_water_availability').reset_index()
wfc_df['lat'] = round(wfc_df['lat'],3)
wfc_df['lon'] = round(wfc_df['lon'],3)


# add coordinate column as id
opt_sw_df['coord'] = np.nan
for i in range(len(opt_sw_df['lat'])):
    print(i)
    opt_sw_df['coord'][i] = str(opt_sw_df['lat'][i])+'_' + str(opt_sw_df['lon'][i])

df_cal['coord'] = np.nan
for i in range(len(df_cal['lat'])):
    print(i)
    df_cal['coord'][i] = str(df_cal['lat'][i])+'_' + str(df_cal['lon'][i])
    
wfc_df['coord'] = np.nan
for i in range(len(wfc_df['lat'])):
    print(i)
    wfc_df['coord'][i] = str(wfc_df['lat'][i])+'_' + str(wfc_df['lon'][i])
    
# join the dfs
reg_df = pd.merge(opt_sw_df, df_cal.loc[:,['sowing_date_cal', 'coord']], on='coord').merge(wfc_df.loc[:,['plant_water_availability', 'coord']], on='coord')

reg_df.dropna(inplace=True) # just for datalines without Nan
# 'lat', 'lon', 'sowing_date', 'coord', 'sowing_date_cal','plant_water_availability'

# model for optimal sowing date
model_opt = smf.ols('sowing_date ~ plant_water_availability', data=reg_df)
res_opt = model_opt.fit()
print(res_opt.summary())
pred_ols_opt = res_opt.get_prediction()
iv_l_opt = pred_ols_opt.summary_frame()["obs_ci_lower"]
iv_u_opt = pred_ols_opt.summary_frame()["obs_ci_upper"]

# model for crop calendar
model_cal = smf.ols('sowing_date_cal ~ plant_water_availability', data=reg_df)
res_cal = model_cal.fit()
# Inspect the results
print(res_cal.summary())
pred_ols_cal = res_cal.get_prediction()
iv_l_cal = pred_ols_cal.summary_frame()["obs_ci_lower"]
iv_u_cal = pred_ols_cal.summary_frame()["obs_ci_upper"]



fig,ax = plt.subplots(figsize=(8, 6))
plt.plot(reg_df['plant_water_availability'], reg_df['sowing_date_cal'], "x", label="Crop Calendar", color='blue')
plt.plot(reg_df['plant_water_availability'], reg_df['sowing_date'], "+", label="Optimal Simulated", color='orange')

#ax.plot(x, y_true, "b-", label="True")
plt.plot(reg_df['plant_water_availability'], res_cal.fittedvalues, color='blue', linestyle='-', label="OLS Crop Calendar")
plt.plot(reg_df['plant_water_availability'], res_opt.fittedvalues, color='orange', linestyle='-', label="OLS Optimal Simulated")
"""
plt.plot(reg_df['plant_water_availability'], iv_u_cal, color='blue', linestyle='dotted', label='upper/lower conf. interval Cal' )
plt.plot(reg_df['plant_water_availability'], iv_l_cal, color='blue', linestyle='dotted')
plt.plot(reg_df['plant_water_availability'], iv_u_opt, color='orange', linestyle='dotted', label='upper/lower conf. interval Opt' )
plt.plot(reg_df['plant_water_availability'], iv_l_opt, color='orange', linestyle='dotted')
"""
#plt.plot([270,500], [270,500], color = 'grey', linestyle = 'dashed', label = '1:1 Line')
plt.xlim((0,160))
plt.xlabel('Plant water availability [mm]', {'fontsize': 11})
plt.ylabel('Crop Calendar/Optimal Simulated sowing dates [doy]', {'fontsize': 11})
plt.suptitle(region, fontsize = 13)
plt.title('Plant water availability vs. Crop Calendar resp. Optimal Simulated sowing dates', {'fontsize': 12})
#plt.text(340, 390, f'R2 = {round(res.rsquared,3)}')
#plt.text(340, 380, f'p value = {round(res.f_pvalue,4)}')
plt.legend(loc="best")



# visualize it

# load political boundaries
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


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

norm = MidpointNormalize(midpoint=0, vmin=0, vmax=250)

fig, ax = plt.subplots()
wfc_da.plot(norm=norm, cmap='RdYlBu')
gdf_all_nat.boundary.plot(ax=ax, linewidth=1, color='black')

#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))
plt.title(region, {'fontsize': 15})
plt.xlabel('Longitude [°E]', {'fontsize': 12})
plt.ylabel('Latitude [°N]', {'fontsize': 12})

plt.savefig('C:/Users/arulrich/Desktop/MA/Writing/Images/soil_type_Zam.png', dpi=260)





