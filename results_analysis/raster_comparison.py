from glob import glob
import numpy as np
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import mapping
import xarray as xr
from itertools import product
import time

from osgeo import gdal
import os
import json

from raster_functions import *
from compile_seasonal_water import check_dirs_exist
from random_forest.compile_classified_points import print_time


INIT_PATH = 'D:/Research/data'

SEASON_LST = ['SPRING', 'SUMMER', 'FALL', 'WINTER']

YEAR_LST = ['2017', '2019']

# from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def reclass_rxr(data_rxr, data_name:str):
    '''
    Reclassify pixel values.
    
    :param data_rxr: xr.DataArray of pixel values
    :param data_name: string indicating which dataset the DataArray is from
    '''
    if data_name == 'planet':
        # classify non-water as 2
        data_rxr = data_rxr.where(((data_rxr > 50) | 
                                (data_rxr == 255)), 2)
        # classify water as 1
        data_rxr = data_rxr.where(((data_rxr <= 50) | 
                                (data_rxr == 255)), 1)
    elif data_name == 'dswe':
        # classify water as 10
        data_rxr = data_rxr.where(((data_rxr <= 0) | 
                                (data_rxr == 255)), 10)
        # classify non-water as 20
        data_rxr = data_rxr.where(((data_rxr == 10) | 
                                (data_rxr == 255)), 20)
    elif data_name == 'gsw':
        # classify water as 100
        data_rxr = data_rxr.where(((data_rxr < 1) | 
                                (data_rxr == 255)), 100)
        # classify non-water as 200
        data_rxr = data_rxr.where(((data_rxr > 0) | 
                                (data_rxr == 255)), 200)
    return data_rxr


def clip_raster(data_rxr, outpath:str, data_name:str):
    '''
    Clip raster to HUC08 boundary and reclassify it.
    
    :param data_rxr: xr.DataArray of pixel values
    :param outpath: string where clipped and reclassified raster is saved
    :param data_name: string indicating which dataset the DataArray is from
    '''

    start_clip = time.time()

    # Check if the clipped and reclassified raster extists.
    # If it does, read in and return the raster
    if check_dirs_exist(outpath) == 'y':
        data_reproj_rxr = rxr.open_rasterio(outpath)
        end_clip = time.time()
        print_time(start_clip, end_clip, f"{os.path.basename(outpath)} exists")
        return data_reproj_rxr

    # Read in the HUC08 shapefile
    shp_reproj_path = f"{INIT_PATH}/Shapefiles/HUC08_03130001.shp"
    
    # Check the shapefile is in the correct projection. Reproject if it is not.
    if check_dirs_exist(shp_reproj_path) == 'y':
        huc8_reproj = gpd.read_file(shp_reproj_path)
    else:
        shp = gpd.read_file(f"{INIT_PATH}/NWI/HU8_03130001_Watershed/HU8_03130001_Watershed.shp")
        huc8_reproj = shp.to_crs(crs=data_rxr.rio.crs)

    # Clip the raster to the HUC08 shapefile
    data_reproj_rxr = data_rxr.astype(float).rio.clip(huc8_reproj.geometry.apply(mapping))
    # Fill in no-data with a value of 255
    data_reproj_rxr = data_reproj_rxr.fillna(255)
    # Reclassify the raster so that water has a value of 1 and non-water has a value of 2 
    # (placement of value --- hundreds, tens, units --- depends on the original raster dataset)
    data_reproj_rxr = reclass_rxr(data_reproj_rxr, data_name)
    # If this is the PlanetBasemap dataset, save the reclassified raster
    if data_name == 'planet':
        data_reproj_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW")

    end_clip = time.time()
    print_time(start_clip, end_clip, f"{os.path.basename(outpath)} clipped")
    
    return data_reproj_rxr

def get_pixel_counts(path:str):
    '''
    Get pixel counts and total area of where each dataset classified water with respect to the other datasets
    
    :param path: string of file path to the raster that combined all three surface water datasets
    '''
    # Get year, season, and confidence information from raster filepath.
    year = path.split('_')[-3]
    szn = path.split('_')[-2]
    conf = path.split('_')[-1][:-4]

    # Read in raster
    rstr = rxr.open_rasterio(path)
    # Get the unique pixel values and the count of pixels with each value from the raster
    pixel_vals, pixel_count = np.unique(rstr, return_counts=True)
    # Get the pixel resolution
    x_res = rstr.rio.resolution()[0]
    y_res = rstr.rio.resolution()[1] 
    # Calculate the area of each unique value based on the pixel count and resoltion
    pixel_area_lst = [i * x_res * -1 * y_res / 1e6 for i in pixel_count] # km2
    # Create a dictionary with pixel values as keys and (pixel count, pixel area) as values
    class_val_dict = dict(zip(pixel_vals, list(zip(pixel_count, pixel_area_lst))))
    # List of all possible pixel values based on the sum of the three reclassified rasters
    possible_keys = [111, 222, 255,                 # all water, non water, no data
                     211, 266,                      # Planet Basemap and DSWE water, GSW non water or no data
                     121, 356,                      # Planet Basemap and GSW water, DSWE non water or no data
                     112, 365,                      # DSWE and GSW water, Planet Basemap non water or no data
                     221, 276, 456, 511,            # Planet Basemap water, DSWE and GSW non water or no data
                     212, 267, 465, 520,            # DSWE water, Planet Basemap and GSW non water or no data
                     122, 357, 375, 610,            # GSW water, Planet Basemap and DSWE non water or no data
                     277, 457, 475, 512, 530, 710]  # all non water or no data
    
    # If there were no pixels for one of the possible values, add that value as a key to the dictionary with pixel count and pixel area as (0,0)
    for i in possible_keys:
        if i not in class_val_dict.keys():
            class_val_dict[i] = (0, 0)
    
    # Create a dictionary that converts possible pixel values into classses related to which datasets classified pixels as water
    class_dict = {'all' : class_val_dict[111],
                  'Planet_DSWE' : [i + j for i, j in zip(class_val_dict[211] , class_val_dict[266])],
                  'Planet_GSW' : [i + j for i, j in zip(class_val_dict[121] , class_val_dict[356])],
                  'DSWE_GSW' : [i + j for i, j in zip(class_val_dict[112] , class_val_dict[365])],
                  'PlanetBasemap' : [i + j + k + l for i, j, k, l in zip(class_val_dict[221], class_val_dict[276], class_val_dict[456], class_val_dict[511])],
                  'DSWE' : [i + j + k + l  for i, j, k, l in zip(class_val_dict[212] , class_val_dict[267], class_val_dict[465], class_val_dict[520])],
                  'GSW' : [i + j + k + l  for i, j, k, l in zip(class_val_dict[122] , class_val_dict[357], class_val_dict[375], class_val_dict[610])]}
    
    print(f'{year}\t{szn} {conf}')
    print("\t", class_dict)
    
    del(rstr)
        
    return class_dict


def main():
    # get pairs of year-szn dswe and planetscope

    yr_szn_class_dict = {}
    for year in YEAR_LST:
        start_year = time.time()
        for szn in SEASON_LST:
            start_szn = time.time()
            for conf in ['conf1', 'conf2', 'conf3']:

                # Check if the raster that combined all three surface water datasets exists
                outpath = f"{INIT_PATH}/Planet_DSWE_GSW/50thresh/planet_dswe50_gsw_{year}_{szn}_{conf}.tif"
                if check_dirs_exist(outpath) == 'y':
                    # If yes, get pixel counts and total area of where each dataset classified water with respect to the other datasets
                    class_dict = get_pixel_counts(outpath)
                    # Add the dictionary of water classified information to a dictionary 
                    # specifying the year, season, and DSWE confidence level of the information
                    yr_szn_class_dict[f'{year}_{szn}_{conf}'] = class_dict
                    end_szn = time.time()
                    print_time(start_szn, end_szn, f"{os.path.basename(outpath)} exists")
                    continue

                # crop planet full tif to study area
                planet_crop_path = f"{INIT_PATH}/PlanetBasemaps/{year}/{szn}/cropped_5070_{year}_{szn}.tif"
                if check_dirs_exist(planet_crop_path) == 'y':
                    planet_rxr = rxr.open_rasterio(planet_crop_path)
                else:
                    planet_path = f"{INIT_PATH}/PlanetBasemaps/{year}/{szn}/full_5070_{year}_{szn}.tif"
                    
                    planet_rxr = rxr.open_rasterio(planet_path)
                    # Clip and reclassify raster
                    planet_rxr = clip_raster(planet_rxr, planet_crop_path, 'planet')

                # resample dswe full tif to match planet
                dswe_resamp_outpath = f"{INIT_PATH}/DSWE/DSWE_SEASONAL_50thresh/{szn}/DSWE50_cropped_5070_{szn}_{year}_{conf}.tif"
                if check_dirs_exist(dswe_resamp_outpath) == 'y':
                    dswe_resamp = rxr.open_rasterio(dswe_resamp_outpath)
                else:
                    dswe_path = f"{INIT_PATH}/DSWE/DSWE_SEASONAL_50thresh/{szn}/DSWEc2_{szn}{year}_50thresh_{conf}.tif"
                    dswe_rxr = rxr.open_rasterio(dswe_path)
                    # Clip and reclassify raster
                    dswe_reclass = clip_raster(dswe_rxr, dswe_resamp_outpath, 'dswe')
                    del(dswe_rxr)
                    # Resample to match PlanetBasemap resolution
                    dswe_resamp = resamp(dswe_reclass, planet_rxr, dswe_resamp_outpath)
                    del(dswe_reclass)
                    del(dswe_resamp)
                    dswe_resamp = rxr.open_rasterio(dswe_resamp_outpath)
                
                # resample gsw full tif to match planet
                gsw_resamp_outpath = f"{INIT_PATH}/GSW/{szn}/GSW_cropped_5070_{szn}_{year}.tif"
                if check_dirs_exist(gsw_resamp_outpath) == 'y':
                    gsw_resamp = rxr.open_rasterio(gsw_resamp_outpath)
                else:
                    gsw_path = f"{INIT_PATH}/GSW/{szn}/GSW_{year}_{szn}.tif"
                    gsw_rxr = rxr.open_rasterio(gsw_path)
                    # Clip and reclassify raster
                    gsw_reclass = clip_raster(gsw_rxr, gsw_resamp_outpath, 'gsw')
                    del(gsw_rxr)
                    # Resample to match PlanetBasemap resolution
                    gsw_resamp = resamp(gsw_reclass, planet_rxr, gsw_resamp_outpath)
                    del(gsw_reclass)
                    del(gsw_resamp)
                    gsw_resamp = rxr.open_rasterio(gsw_resamp_outpath)
                # new raster values 
                # water: planet = 1, dswe = 10, gsw = 100
                # non-water: planet = 2, dswe = 20, gsw = 200
                # 
                # add rasters together (concurrance of water = 111, non-water = 222)
                stacked_rxr = xr.concat([planet_rxr, dswe_resamp, gsw_resamp], 
                                        dim='band').assign_coords({'band':['planet', 'dswe', 'gsw']})
                del(planet_rxr, dswe_resamp, gsw_resamp)
                stacked_rxr = xr.concat([stacked_rxr, 
                                        (stacked_rxr[0] + stacked_rxr[1] + stacked_rxr[2]).assign_coords({'band':'planet_dswe_gsw'})], 
                                        dim='band')
                
                stacked_rxr[3] = stacked_rxr[3].where(stacked_rxr[3] != 765, 255)

                # Save raster that combined all three surface water datasets 
                stacked_rxr[3].rio.to_raster(outpath, driver="GTiff", compress="LZW")

                del(stacked_rxr)

                end_year = time.time()
                print_time(start_year, end_year, f"{os.path.basename(outpath)} saved")
                
                # save class count info
                class_dict = get_pixel_counts(outpath)
                # Add the dictionary of water classified information to a dictionary 
                # specifying the year, season, and DSWE confidence level of the information
                yr_szn_class_dict[f'{year}_{szn}_{conf}'] = class_dict
                end_szn = time.time()
                print_time(start_szn, end_szn, f"{os.path.basename(outpath)} exists")

    # Save dictionary with all water classification information for all years, seasons, and DSWE confidence levels
    with open(f"{INIT_PATH}/Planet_DSWE_GSW/50thresh/class_counts_50thresh.json", "w") as outfile:
        json.dump(yr_szn_class_dict, outfile, indent=4, cls=NpEncoder)

    return


if __name__ == '__main__':
    main()
