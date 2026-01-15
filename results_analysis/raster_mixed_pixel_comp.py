from glob import glob
import numpy as np
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import mapping
import xarray as xr
from itertools import product
import time

from osgeo import gdal
import pandas as pd
import os
import json

from raster_functions import *
from compile_seasonal_water import check_dirs_exist
from random_forest.compile_classified_points import print_time

from rasterio.enums import Resampling

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

#####################################################################################################################
###                                                                                                               ###
###             Issue with resamp. Can't get resamp to return as float. Switched to QGIS for sake of time
###                                                                                                               ###
#####################################################################################################################

def resamp(srtm_rxr:xr.core.dataarray.DataArray, data_rxr:xr.core.dataarray.DataArray, outpath:str):
    '''
        Reproject and resample the SRTM data (elevation, slope, or hillshade) to match the
        Planet Basemap data
    '''
    # reproject, resample
    resamp_rxr1 = srtm_rxr.rio.reproject_match(match_data_array=data_rxr, resampling=Resampling.average , nodata=255, type=float)
    resamp_rxr = resamp_rxr1.assign_coords({'x':data_rxr.x,
                                            'y':data_rxr.y})
    # save reprojected/resampled raster
    resamp_rxr = resamp_rxr.rio.write_nodata(255, inplace=True)
    resamp_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW", dtype="float32")
    print(f'{os.path.basename(outpath)} saved.')
    # return the DataArray
    return resamp_rxr

def reclass_rxr(data_rxr, data_name):
    if data_name == 'planet':
        # classify non-water as 2
        data_rxr = data_rxr.where(((data_rxr > 50) | 
                                (data_rxr == 255)), 255)
        # classify water as 1
        data_rxr = data_rxr.where(((data_rxr <= 50) | 
                                (data_rxr == 255)), 1)
    elif data_name == 'dswe':
        # classify water as 10
        data_rxr = data_rxr.where(((data_rxr <= 0) | 
                                (data_rxr == 255)), 10)
        # classify non-water as 20
        data_rxr = data_rxr.where(((data_rxr == 10) | 
                                (data_rxr == 255)), 255)
    elif data_name == 'gsw':
        # classify water as 100
        data_rxr = data_rxr.where(((data_rxr < 1) | 
                                (data_rxr == 255)), 100)
        # classify non-water as 200
        data_rxr = data_rxr.where(((data_rxr > 0) | 
                                (data_rxr == 255)), 255)
    return data_rxr


def clip_raster(data_rxr, outpath, data_name):

    start_clip = time.time()

    if check_dirs_exist(outpath) == 'y':
        data_reproj_rxr = rxr.open_rasterio(outpath)
        end_clip = time.time()
        print_time(start_clip, end_clip, f"{os.path.basename(outpath)} exists")
        return data_reproj_rxr

    shp_reproj_path = f"{INIT_PATH}/Shapefiles/HUC08_03130001.shp"
    
    if check_dirs_exist(shp_reproj_path) == 'y':
        huc8_reproj = gpd.read_file(shp_reproj_path)
    else:
        shp = gpd.read_file(f"{INIT_PATH}/NWI/HU8_03130001_Watershed/HU8_03130001_Watershed.shp")
        huc8_reproj = shp.to_crs(crs=data_rxr.rio.crs)

    data_reproj_rxr = data_rxr.astype(float).rio.clip(huc8_reproj.geometry.apply(mapping))
    data_reproj_rxr = data_reproj_rxr.fillna(255)
    data_reproj_rxr = reclass_rxr(data_reproj_rxr, data_name)
    if data_name == 'dswe':
        print(data_rxr.rio.crs)
        data_reproj_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW")

    end_clip = time.time()
    print_time(start_clip, end_clip, f"{os.path.basename(outpath)} clipped")
    
    return data_reproj_rxr


def main():
    # get pairs of year-szn dswe and planetscope

    for conf in ['conf1', 'conf2', 'conf3']:

        for szn in SEASON_LST:
            start_szn = time.time()

            for year in YEAR_LST:
                start_year = time.time()

                # check we have cropped dswe
                dswe_crop_outpath = f"{INIT_PATH}/DSWE/DSWE_SEASONAL_50thresh/{szn}/30m/DSWE_cropped_5070_{szn}_{year}_{conf}_30m_avg.tif"
                if check_dirs_exist(dswe_crop_outpath) == 'y':
                    dswe_rxr = rxr.open_rasterio(dswe_crop_outpath)
                else:
                    dswe_path = f"{INIT_PATH}/DSWE/DSWE_SEASONAL_50thresh/{szn}/DSWEc2_{szn}{year}_50thresh_{conf}.tif"
                    dswe_rxr = rxr.open_rasterio(dswe_path)
                    dswe_rxr = clip_raster(dswe_rxr, dswe_crop_outpath, 'dswe')                
                
                # resample planet cropped tif to match dswe
                planet_30m_path = f"{INIT_PATH}/PlanetBasemaps/{year}/{szn}/30m/cropped_5070_{year}_{szn}_30m_avg.tif"
                if check_dirs_exist(planet_30m_path) == 'y':
                    planet_resamp = rxr.open_rasterio(planet_30m_path)
                else:
                    print(f'{os.path.basename(planet_30m_path)} missing')
                    # planet_path = f"{INIT_PATH}/PlanetBasemaps/{year}/{szn}/full_5070_{year}_{szn}.tif"
                    # planet_rxr = rxr.open_rasterio(planet_path)

                    # planet_reclass = clip_raster(planet_rxr, planet_30m_path, 'planet')
                    # del(planet_rxr)
                    # planet_resamp = resamp(planet_reclass, dswe_rxr, planet_30m_path)
                    # del(planet_reclass)
                    # del(planet_resamp)
                    # planet_resamp = rxr.open_rasterio(planet_30m_path)
                
                # Mask out DSWE water from the resampled Planet
                planet_mask = planet_resamp.where(dswe_rxr != 10)

                # 0.00 < px < 1.00
                mask_1 = (planet_mask > 0) & (planet_mask < 1) & (planet_mask != 255)
                mask_1_val = planet_mask.where(mask_1).count().values

                # 0.00 < px <= 0.25
                mask_2 = (planet_mask > 0) & (planet_mask <= 0.25) & (planet_mask != 255)
                mask_2_val = planet_mask.where(mask_2).count().values
                
                # 0.25 < px <= 0.50
                mask_3 = (planet_mask > 0.25) & (planet_mask <= 0.5) & (planet_mask != 255)
                mask_3_val = planet_mask.where(mask_3).count().values

                # 0.50 < px <= 0.75
                mask_4 = (planet_mask > 0.5) & (planet_mask <= 0.75) & (planet_mask != 255)
                mask_4_val = planet_mask.where(mask_4).count().values

                # 0.75 < px < 1.00
                mask_5 = (planet_mask > 0.75) & (planet_mask < 1.0) & (planet_mask != 255)
                mask_5_val = planet_mask.where(mask_5).count().values

                # px == 1
                mask_6 = (planet_mask == 1) & (planet_mask != 255)
                mask_6_val = planet_mask.where(mask_6).count().values

                print(f'\n\n{year} {szn} --- {conf}')
                print(f'pixel counts:')
                print(f'\t{mask_1_val}')
                print(f'\t{mask_2_val}')
                print(f'\t{mask_3_val}')
                print(f'\t{mask_4_val}')
                print(f'\t{mask_5_val}')
                print(f'\t{mask_6_val}')

                print(f'\nk2m:')
                print(f'\t{mask_1_val *30*30 / 1e6}')
                print(f'\t{mask_2_val *30*30 / 1e6}')
                print(f'\t{mask_3_val *30*30 / 1e6}')
                print(f'\t{mask_4_val *30*30 / 1e6}')
                print(f'\t{mask_5_val *30*30 / 1e6}')
                print(f'\t{mask_6_val *30*30 / 1e6}')

    return


if __name__ == '__main__':
    main()
