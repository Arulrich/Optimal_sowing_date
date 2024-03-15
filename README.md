# Optimal_sowing_date
In this repository, you find the Python codes I used for the data analysis of my master's thesis `The effect of sowing date sources on yield estimates of spatialized crop growth models in Sub-Saharan Africa`.
The first three codes have to be run one after the other (first 1) then 2) then 3)) and are for identifying the optimal simulated sowing dates. The fourth is for their visualization. The other codes are for comparison with the other sowing date sources (Crop Calendar and Observations). 
For each code, I briefly explain what they do and => what are the output.

1) ensemble.py
Build a dataarray of simulated yield in each location, time, and sowing date of the ensemble of models and of the single models as well.
Build a dataarray of yield's mean and CV for each sowing date of the ensemble of models and of the single models as well. 
ATTENTION initialize the datasets accordingly to the bounding box area (for now initialize with a sowing date's dataarray with full extent)
 => output/{model}_yield_MgtMais0_allSowingDates_2.0.nc,
		output/{model}_yieldMean_MgtMais0_allSowingDates_2.0.nc, output/{model}_yieldCV_MgtMais0_allSowingDates_2.0.nc

2) opt_sw_main.py
Build a dataframe with the optimal sowing date of the main season from the ensemble of the models or for the single models.
Build a dataframe with the optimal sowing date of the main and the small season if there is any (still to develop, not in the master thesis)
 => output/{model}_MgtMais0_optSwDateMain_2.0.csv

3) df_to_da.py
Convert the csv file into a nc file (dataarray)
 => output/{model}_MgtMais0_optSwDateMain_2.0.nc'

4) visualize_opt_sw.py
Visualize the dataarray with the optimal sowing date and shp files of political boundaries.
	- extraction of yield's mean and cv along sowing dates for single locations
	- visualize single locations
	- decide the possible start of the season
	- shift the doy accordingly
	- visualize optimal sowing dates for the entire area with administrative borders
 => determine the doy shift, various plots (heatmap)

5) comp_calendar.py
Comparison between crop calendar and our optimal sowing date
	- same visualization after disaggregation of crop calendar
	- difference optimal sowing dates - crop calendar
	- absolute differences
	- regression
	- same visualization after aggregation of optimal sowing date (TO DO BETTER)
 => various plots, disaggregated crop calendar: data_analysis/Zim_crop_calendar_0_05.nc

6) comp_obs_Zam.py
Comparison with the observations of Zimbabwe
	- translate the observed sowing dates in doy
	- interpolated optimal simulated and crop calendar sowing dates for every observed location
	- extract the simulated yield value for each location, year, and sowing date.
	- visualize where are the farms in comparison with the crop mask
	- regression between simulated and observed sowing dates
	- regression between sowing date from crop calendar and from observations
	- visualize observed sowing dates, optimal sowing dates and crop calendar, and differences (also absolute) (density plot, boxplot, heatmap; also differentiating the kind of farm)
 => various plots, dataframe with survey and interpolated optimal simulated and crop calendar sowing dates. data_analysis/Zam_Celsius_Dssat_Stics_outputs_90/planting_dates_zam_and_our_data.csv

7) comp_obs_Zim.py
Comparison with the observations Zambia 
	- translate the observed sowing dates in doy
	- interpolated optimal simulated and crop calendar sowing dates for every observed location
	- extract the simulated yield value for each location, year, and sowing date.
	- visualize where are the farms in comparison with the crop mask
	- regression between simulated and observed sowing dates
	- regression between sowing date from crop calendar and from observations
	- visualize observed sowing dates, optimal sowing dates and crop calendar, and differences (also absolute) (density plot, boxplot, heatmap)
 => various plots, dataframe with survey and interpolated optimal simulated and crop calendar sowing dates. data_analysis/Zim_Celsius_Dssat_Stics_outputs_90/planting_dates_zim_2021_and_our_data.csv 


8) yield_extraction.py
Extract 30 years yield mean and cv for each location simulated with the optimal simulated and resp. the crop calendar sowing dates.
	- Visualize them
	- Visualize the difference
 => dataarray like: /output/ensemble_yieldMean_MgtMais0_optSowingDates_2.0.nc, output/ensemble_yieldMean_MgtMais0_Crop_Cal_0_05_SowingDates.nc, output/ensemble_Diff_yieldMean_opt_CropCal_SowingDates.nc, ...

9) analysis_with_shp.py
Use shp file of AEZ or political borders in the analysis: visualization of the dataarray of sowing date or yield, clip of dataarray and extraction of stats (describe(), boxplot, ...)
 => boxplot and stats of sowing_date/yield values per shp region

10) MAPSPAM.py
Assess the average maize production by multiplying the averaged yields simulated with the optimal simulated sowing dates and resp. with the crop calendar sowing dates and the actual physical area of rainfed maize from MAPSPAM.
	- visualization of the absolute and relative differences between the production estimates
 => netcdf file containing maize production values in t at 0.083333° resolution, gpd files of countries containing production estimates aggregated at the province level.

11) soil_type.py
Regression analysis to investigate whether there is a correlation between the root zone total plant available water holding capacity and the sowing dates.
 => visualization of the root zone total plant available water holding capacity in the four areas, regression test and scatterplots.

12) plot_sw_AEZ_swSource.py
Analyze the sowing dates with a violinplot for each sowing date source * AEZ combination.
=> dataframe with columns: latitude, longitude, sowing date (doy), source (crop calendar or optimal simulated), and AEZ (e.g. Tropical_warm_semiarid, ...); violinplot to visualize it. e.g. /WA_sw_per_source_AEZ.csv

13) plot_yield_AEZ_swSource.py
Analyze the simulated yields with a violinplot for each sowing date source * AEZ combination.
=> dataframe with columns: latitude, longitude, yield (t ha-1), source (crop calendar or optimal simulated), and AEZ (e.g. Tropical_warm_semiarid, ...); boxplot to visualize it. e.g. /WA_yield_per_source_AEZ.csv
