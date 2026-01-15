from glob import glob
import rioxarray as rxr
import numpy as np
import xarray as xr
import os
import json
import time
import geopandas as gpd
import pandas as pd
import concurrent.futures
from rasterstats import zonal_stats
from random_forest.compile_classified_points import print_time
import rasterio as rio

from cli import parse_compile

INIT_PATH = 'D:/Research/data'

QUAD_NAME_LIST = ['547-1235', '548-1235', 
                  '545-1234', '546-1234', '547-1234', '548-1234', '549-1234', 
                  '545-1233', '546-1233', '547-1233', '548-1233', '549-1233', 
                  '545-1232', '546-1232', '547-1232', '548-1232', 
                  '544-1231', '545-1231', '546-1231', '547-1231', 
                  '543-1230', '544-1230', '545-1230', '546-1230', 
                  '542-1229', '543-1229', '544-1229', '545-1229', 
                  '543-1228', '544-1228', '545-1228']

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



def check_dirs_exist(path):
    '''
    Make sure file does not already exist and make sure all dirs exist.
    '''
    if os.path.exists(path):
        print(f"{path} exists")
        return 'y'
    elif not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return 'n'

def _get_stacked_quad(file_lst):
    '''
    Stack all quad rasters for a specified year and season to be used in generating a composite.
    The alpha stack is the stack of no-data masks.
    
    :param file_lst: List of classified PlanetBasemap images to be composited
    '''
    for i in range(len(file_lst)):
        rxr_data = rxr.open_rasterio(file_lst[i])
        if i == 0:
            stacked_data_rxr = rxr_data[0]
            stacked_alpha_rxr = rxr_data[1]
        else:
            stacked_data_rxr = xr.concat([stacked_data_rxr, rxr_data[0]], dim='band')
            stacked_alpha_rxr = xr.concat([stacked_alpha_rxr, rxr_data[1]], dim='band')

    return stacked_data_rxr, stacked_alpha_rxr

def _get_percent_water_array(stacked_data_rxr):
    '''
    Calculate a percentage classified as water raster.
    
    :param stacked_data_rxr: Stacked array of all PlanetBasemap classified images for the year, season, and quad
    '''

    # Count the frequency each pixel was classified as water (1)
    sw_sum = np.count_nonzero(stacked_data_rxr==1,axis=0)
    sw_sum = sw_sum.astype('uint8')

    # Count the frequency each pixel was able to be classified (i.e., was valid and not NO DATA)
    non_na_sum = np.count_nonzero(stacked_data_rxr>=0,axis=0)
    non_na_sum = non_na_sum.astype('uint8')

    out = np.zeros(non_na_sum.shape)
    # Calculate the percentage a pixel was classified as WATER out of all valid classificaitons
    p_sw = np.round(np.divide(sw_sum, non_na_sum, out=out, where=non_na_sum!=0) * 100,0)
    p_sw = p_sw.astype('uint8')

    return p_sw

def _save_percent_water_quad(p_sw, stacked_data_rxr, outpath:str):
    '''
    Save the composite raster.
    
    :param p_sw: np.array where each pixel is the percent of times it was classified as surface water
    :param stacked_data_rxr: Stacked array of all PlanetBasemap classified images for the year, season, and quad
    :param outpath: Outpath where the composite raster will be saved
    '''

    # Convert from np.array to xr.DataArray with spatial information
    x_coords = stacked_data_rxr.coords['x'].values
    y_coords = stacked_data_rxr.coords['y'].values
    spatial_coords = stacked_data_rxr.coords['spatial_ref']
    p_sw_rxr = xr.DataArray(p_sw,
                            coords={'x': x_coords, 'y': y_coords, 'spatial_ref': spatial_coords},
                            dims=['y', 'x'])
    # Save the composite raster
    p_sw_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW")
    
    print(f"{outpath} saved")

    del(p_sw)
    del(stacked_data_rxr)

    return

def _get_na_counts(stacked_alpha_rxr, quad_name:str, na_pixel_dict:dict, year_season:str):
    '''
    Get the count and percentage of No Data pixels in the stack of PlanetBasemap images for a quad.
    
    :param stacked_alpha_rxr: DataArray of alpha band (no-data mask) PlanetBasemap data
    :param quad_name: Quad ID / Name
    :param na_pixel_dict: Dictionary of No Data values
    :param year_season: Year-Season string
    '''

    # Get a count of No Data pixels
    na_sum = np.count_nonzero(stacked_alpha_rxr==-9999,axis=0)
    na_sum = na_sum.astype('uint8')

    total_pixels = 4096*4096*13
    na_count_dict = dict(zip(np.unique(na_sum, return_counts=True)[0], np.unique(na_sum, return_counts=True)[1]))
    na_running_sum = 0
    for key in na_count_dict:
        val = na_count_dict[key]
        na_running_sum += key * val

    # Add No Data pixel count and percentage to the dictionary with the year, season, and quad ID as the key
    na_pixel_dict[year_season + '_' + quad_name] = {'na_sum' : na_running_sum,
                                                    'na_percent': na_running_sum / total_pixels}

    return na_pixel_dict

def composite(file_lst:list, na_pixel_dict:dict, year_season:str):
    '''
    Saves seasonal composite of classified PlanetBasemap images.
    
    :param file_lst: List of classified PlanetBasemap filepaths
    :param na_pixel_dict: Dictionary of No Data pixel counts
    :param year_season: Year-Season string
    '''

    start_comp = time.time()

    # Get the name of the quad and the path where the composite raster will be saved.
    quad_name = os.path.basename(file_lst[0]).split('_')[0]
    outpath = os.path.join(os.path.dirname(os.path.dirname(file_lst[0])), 
                        'composites',
                        quad_name + '_composite.tif')

    # Stack all classified PlanetBasemap images for the quad
    stacked_data_rxr, stacked_alpha_rxr = _get_stacked_quad(file_lst)

    # Check if the composite file exists
    if check_dirs_exist(outpath) == 'n':
        # Calculate the percent of times each pixel was classified as water
        p_sw = _get_percent_water_array(stacked_data_rxr)
        # Save the composite raster
        _save_percent_water_quad(p_sw, stacked_data_rxr, outpath)

    # Add the No Data info for the quad to the No Data dictionary
    na_pixel_dict = _get_na_counts(stacked_alpha_rxr, quad_name, na_pixel_dict, year_season)

    end_comp = time.time()
    print_time(start_comp, end_comp, 'composite')

    return na_pixel_dict

def dswe_composite(dswe_path:str, year:str, conf_level:int):
    '''
    Saves seasonal composite of DSWE rasters at various confidence levels.
    
    :param dswe_path: Filepath to DSWE file
    :param year: Year of DSWE classification
    :param conf_level: Minimum DSWE confidence level (1, 2, or 3)
    '''
    start_comp = time.time()

    ########### For Walker 2025 DSWE ###########
    dswe_rxr = rxr.open_rasterio(dswe_path)

    if year == 2017:
        season_dict = {'SPRING': 12, 'SUMMER': 9, 'FALL': 6, 'WINTER':3}
    elif year == 2019:
        season_dict = {'SPRING': 13, 'SUMMER': 10, 'FALL': 7, 'WINTER':4}

    for szn in ['SPRING', 'SUMMER', 'FALL', 'WINTER']:
        outpath = f'D:/Research/data/DSWE/{szn}/DSWEc2_{szn}_{year}_conf{conf_level}.tif'
        if check_dirs_exist(outpath) == 'y':
            return
        
        idx = season_dict[szn]
        if conf_level == 1: # high conf only
            sw_sum = np.count_nonzero(dswe_rxr[idx-3:idx]==1, axis=0) 
        elif conf_level == 2: # high and mod conf
            sw_sum = np.count_nonzero((dswe_rxr[idx-3:idx]==1) | (dswe_rxr[idx-3:idx]==2), axis=0) 
        elif conf_level == 3: # high, mod, and partial conf
            sw_sum = np.count_nonzero((dswe_rxr[idx-3:idx]>=1) & (dswe_rxr[idx-3:idx]<=3), axis=0) 

        # Convert np.array to xr.DataArray
        x_coords = dswe_rxr.coords['x'].values
        y_coords = dswe_rxr.coords['y'].values
        spatial_coords = dswe_rxr.coords['spatial_ref']
        sw_rxr = xr.DataArray(sw_sum,
                                coords={'x': x_coords, 'y': y_coords, 'spatial_ref': spatial_coords},
                                dims=['y', 'x'])
        
        # Save as raster with ESPG:5070
        sw_5070_rxr = sw_rxr.rio.reproject('EPSG:5070', resolution=(30,30)).astype(int)
        sw_5070_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW")

    end_comp = time.time()
    print_time(start_comp, end_comp, 'dswe composite')

    return


def main():

    args = parse_compile()

    dataset = args.dataset

    if dataset == 'ps':
        #####################################################################################
        ######                        Planet Basemaps Compile                         #######
        #####################################################################################
        glob_path_lst = []
        for YEAR in YEAR_LST:
            for SEASON in SEASON_LST:
                glob_path_lst.append([f"{INIT_PATH}/PlanetBasemaps/{YEAR}/{SEASON}/*/*/{i}_classified.tif" \
                                for i in QUAD_NAME_LIST])

        na_pixel_dict = {}     
        for path_lst in glob_path_lst:
            for glob_path in path_lst:
                year_season = glob_path.split('/')[4] + '-' + glob_path.split('/')[5]
                file_lst = glob(glob_path)

                na_pixel_dict = composite(file_lst, na_pixel_dict, year_season)

        json_obj = json.dumps(na_pixel_dict, indent=4, cls=NpEncoder)
        # Writing to sample.json
        with open(f"{INIT_PATH}/PlanetBasemaps/na_pixel_info.json", "w") as outfile:
            outfile.write(json_obj)

    elif dataset == 'dswe':
        #####################################################################################
        ######                        Walker GEE DSWE Compile                         #######
        #####################################################################################

        for year in [2017, 2019]:
            dswe_path = f'{INIT_PATH}/DSWE_SE/Walker_GEE_DSWE/Walker_GEE_DSWEc2_{year}_months.tif'
            
            dswe_composite(dswe_path, year, 1) # high conf only
            dswe_composite(dswe_path, year, 2) # high and mod conf
            dswe_composite(dswe_path, year, 3) # high, mod, and partial conf


    return


if __name__ == '__main__':
    main()

