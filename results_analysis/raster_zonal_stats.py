from glob import glob
import rioxarray as rxr
from rioxarray.merge import merge_arrays
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
from compile_seasonal_water import check_dirs_exist

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


def merge_planet_composites(fldr:str):
    '''
    Merge list of classified PlanetBasemap rasters into a single raster.
    
    :param fldr: String of the directory containing the seasonally composited PlanetBasemap rasters.
    '''

    start_merge = time.time()

    tif_path_lst = glob(os.path.join(fldr,'*.tif'))

    year = tif_path_lst[0].split('/')[4]
    szn = tif_path_lst[0].split('/')[5]
    outpath = f"{INIT_PATH}/PlanetBasemaps/{year}/{szn}/full_5070_{year}_{szn}.tif"
    if check_dirs_exist(outpath) == 'y':
        end_merge = time.time()
        print_time(start_merge, end_merge, f'{os.path.basename(outpath)} exists')
        return
    
    rxr_lst = []
    for i in range(len(tif_path_lst)):
        rxr_data = rxr.open_rasterio(tif_path_lst[i])
        rxr_lst.append(rxr_data)

    rxr_merged = merge_arrays(rxr_lst)

    rxr_merged_5070 = rxr_merged.rio.reproject('EPSG:5070', resolution=(4.77, 4.77)).astype(int)
    rxr_merged_5070.rio.to_raster(outpath, driver="GTiff", compress="LZW")

    end_merge = time.time()
    print_time(start_merge, end_merge, f'merged {os.path.basename(outpath)}')

    del(rxr_data, rxr_lst, rxr_merged, rxr_merged_5070)
    
    return


def process_huc12_zonal_stats(fldr:str, data_name:str):
    '''
    Calculate zonal statistics of satellite surface water classifications 
     on HUC12 subwatersheds
    
    :param fldr: Folder containing classified rasters
    :param data_name: Name of the satellite imagery dataset (PlanetBasemap, DSWE, DSWE50, GSW)
    '''
    start_fldr = time.time()

    # Get year and season info
    if data_name == 'PlanetBasemaps':
        yr_lst = [fldr.split('/')[4]]
        szn = fldr.split('/')[5]

    elif data_name == 'DSWE':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[4]
    
    elif data_name == 'DSWE50':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[5]

    elif data_name == 'GSW':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[4]

    # Get list of rasters.
    tif_paths = glob(os.path.join(fldr,'*crop*.tif'))
    tif_paths.sort()

    # Loop through each raster
    for i in range(len(tif_paths)):

        # Set path to CSV where zonal stats data will be saved
        if 'DSWE' in data_name:
            yr = tif_paths[i].split('_')[-2]
            conf = tif_paths[i].split('_')[-1][:-4]
            out_path = os.path.join(f'{INIT_PATH}/huc_stats_p3/{data_name}/{yr}_{szn}_{data_name}_{conf}.csv')

        else:
            yr = yr_lst[i]
            out_path = os.path.join(f'{INIT_PATH}/huc_stats_p3/{data_name}/{yr}_{szn}_{data_name}.csv')
        # make sure outpath dirs exist

        print(yr, szn, 'in process', os.getpid())

        if check_dirs_exist(out_path) == 'y':
            return
        
        start_tif = time.time()
        # Open raster
        src = rio.open(tif_paths[i])
        # Read in HUC12 shapefile
        shp = gpd.read_file(f'{INIT_PATH}/NWI/HU8_03130001_Watershed/HUC12_03130001/HUC12_03130001.shp')
        # Reproject shapeile to match raster
        huc12_reproj = shp.to_crs(crs=src.crs.data)
        del(shp)
        # Select columns of interest
        huc12 = huc12_reproj[['huc12','geometry']]
        del(huc12_reproj)
        # Initialize a pd.DataFrame
        huc12_df = pd.DataFrame(huc12['huc12'])
        huc12_df['total_water'] = 0
        huc12_df['total_pixels'] = 0
        res_x = src.res[0]
        res_y = src.res[1]

        # Read in raster
        array = src.read(1)
        affine = src.transform

        # Reclassify raster so water = 1 and non-water = 0
        if 'DSWE' in data_name:
            water_array = np.where((array >= 20) & (array != 255), 0, array)
            water_array = np.where((water_array == 10) & (water_array != 255), 1, water_array)
            # water_array = (array > 0).astype('uint8')

        elif data_name == 'PlanetBasemaps':
            water_array = np.where((array >= 2) & (array != 255), 0, array)
            water_array = np.where((water_array == 1) & (water_array != 255), 1, water_array)

        elif data_name == 'GSW':
            water_array = np.where((array >= 200) & (array != 255), 0, array)
            water_array = np.where((water_array == 100) & (water_array != 255), 1, water_array)
            
        del(array)

        # Calculate the sum of pixels and the count of pixels within the HUC12 polygons
        zs = zonal_stats(huc12, water_array, affine=affine, stats=['sum', 'count'], nodata=255, all_touched = False)

        del(water_array)

        # Put zonal stats into a dataframe
        zs_df = pd.DataFrame(zs)
        zs_df = zs_df.fillna(0)

        # Add the zonal stats information to the huc12 dataframe
        huc12_df['total_water'] += zs_df['sum']
        huc12_df['total_pixels'] += zs_df['count']

        # Add pixel resoltion columns to the dataframe
        huc12_df['res_x'] = res_x
        huc12_df['res_y'] = res_y
        # Save dataframe as a csv
        huc12_df.to_csv(out_path, index=False)

        # Close the raster
        src.close()
        end_tif = time.time()
        print_time(start_tif, end_tif, szn + os.path.basename(tif_paths[i]))


    end_fldr = time.time()
    print_time(start_fldr, end_fldr, yr+szn+'FINISHED!')
    return

def param_huc12_wrapper(p):
    return process_huc12_zonal_stats(*p)


def process_nwi_zonal_stats(fldr:str, data_name:str):
    '''
    Calculate zonal statistics of satellite surface water classifications 
     on National Wetland Inventory water bodies
    
    :param fldr: Folder containing classified rasters
    :param data_name: Name of the satellite imagery dataset (PlanetBasemap, DSWE, DSWE50, GSW)
    '''
    start_fldr = time.time()

    # Get year and season info
    if data_name == 'PlanetBasemaps':
        yr_lst = [fldr.split('/')[4]]
        szn = fldr.split('/')[5]

    elif data_name == 'DSWE':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[4]

    elif data_name == 'DSWE50':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[5]

    elif data_name == 'GSW':
        yr_lst = [2017, 2019]
        szn = fldr.split('/')[4]

    # Get list of rasters.
    tif_paths = glob(os.path.join(fldr,'*crop*.tif'))
    tif_paths.sort()

    # Loop through each raster
    for i in range(len(tif_paths)):

        # Set path to CSV where zonal stats data will be saved
        if 'DSWE' in data_name:
            yr = tif_paths[i].split('_')[-2]
            conf = tif_paths[i].split('_')[-1][:-4]
            out_path = os.path.join(f'{INIT_PATH}/nwi_stats_p3/{data_name}/{yr}_{szn}_{data_name}_{conf}.csv')

        else:
            yr = yr_lst[i]
            out_path = os.path.join(f'{INIT_PATH}/nwi_stats_p3/{data_name}/{yr}_{szn}_{data_name}.csv')
        # make sure outpath dirs exist

        print(yr, szn, 'in process', os.getpid())

        if check_dirs_exist(out_path) == 'y':
            continue
        
        start_tif = time.time()
        # Open raster
        src = rio.open(tif_paths[i])
        # Read in HUC08 National Wetland Inventory LRP (Lakes, Rivers, Ponds) shapefile
        shp = gpd.read_file(f'{INIT_PATH}/NWI/HU8_03130001_Watershed/NWI_LRP.shp')
        # Reproject shapeile to match raster
        nwi_reproj = shp.to_crs(crs='EPSG:5070')
        del(shp)
        # Select columns of interest
        nwi = nwi_reproj[['WETLAND_TY', 'layer','geometry']]
        del(nwi_reproj)
        # Initialize a pd.DataFrame
        nwi_df = pd.DataFrame(nwi['WETLAND_TY'])
        nwi_df['total_water'] = 0
        nwi_df['total_pixels'] = 0
        res_x = src.res[0]
        res_y = src.res[1]

        # Read in raster
        array = src.read(1)
        affine = src.transform

        # Reclassify raster so water = 1 and non-water = 0
        if 'DSWE' in data_name:
            water_array = np.where((array >= 20) & (array != 255), 0, array)
            water_array = np.where((water_array == 10) & (water_array != 255), 1, water_array)
            # water_array = (array > 0).astype('uint8')

        elif data_name == 'PlanetBasemaps':
            water_array = np.where((array >= 2) & (array != 255), 0, array)
            water_array = np.where((water_array == 1) & (water_array != 255), 1, water_array)

        elif data_name == 'GSW':
            water_array = np.where((array >= 200) & (array != 255), 0, array)
            water_array = np.where((water_array == 100) & (water_array != 255), 1, water_array)
            
        del(array)

        # Calculate the sum of pixels and the count of pixels within the NWI LRP polygons
        zs = zonal_stats(nwi, water_array, affine=affine, stats=['sum', 'count'], nodata=255, all_touched = False)

        del(water_array)

        # Put zonal stats into a dataframe
        zs_df = pd.DataFrame(zs)
        zs_df = zs_df.fillna(0)

        # Add the zonal stats information to the NWI dataframe
        nwi_df['total_water'] += zs_df['sum']
        nwi_df['total_pixels'] += zs_df['count']

        # Add pixel resoltion columns to the dataframe
        nwi_df['res_x'] = res_x
        nwi_df['res_y'] = res_y
        # Save dataframe as a csv
        nwi_df.to_csv(out_path, index=False)

        # Close the raster
        src.close()
        end_tif = time.time()
        print_time(start_tif, end_tif, szn + os.path.basename(tif_paths[i]))


    end_fldr = time.time()
    print_time(start_fldr, end_fldr, yr+szn+'FINISHED!')
    return

def param_nwi_wrapper(p):
    return process_nwi_zonal_stats(*p)



def main():

    #####################################################################################
    ######                       Planet Basemaps Zonal Stats                      #######
    #####################################################################################
    fldr_lst = []
    for YEAR in YEAR_LST:
        for SEASON in SEASON_LST:
            fldr_lst.append(f"{INIT_PATH}/PlanetBasemaps/{YEAR}/{SEASON}/composites/")

    for fldr in fldr_lst:
        merge_planet_composites(fldr)

    fldr_lst = []
    for YEAR in YEAR_LST:
        for SEASON in SEASON_LST:
            fldr_lst.append(f"{INIT_PATH}/PlanetBasemaps/{YEAR}/{SEASON}/")


    params = ((fldr, 'PlanetBasemaps') for fldr in fldr_lst)

    # run HUC12 zonal stats in 4 processes
    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=4
    # ) as executor:
    #     executor.map(param_huc12_wrapper, params)

    # run NWI zonal stats in 4 processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4
    ) as executor:
        executor.map(param_nwi_wrapper, params)

    #####################################################################################
    ######                      Walker GEE DSWE Zonal Stats                       #######
    #####################################################################################

    fldr_lst = []
    # for YEAR in YEAR_LST:
    for SEASON in SEASON_LST:
        fldr_lst.append(f"{INIT_PATH}/DSWE/DSWE_SEASONAL_50thresh/{SEASON}/")

    params = ((fldr, 'DSWE50') for fldr in fldr_lst)

    # run HUC12 zonal stats in 4 processes
    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=4
    # ) as executor:
    #     executor.map(param_huc12_wrapper, params)

    # run NWI zonal stats in 4 processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4
    ) as executor:
        executor.map(param_nwi_wrapper, params)

    #####################################################################################
    ######                          GSW DSWE Zonal Stats                          #######
    #####################################################################################

    fldr_lst = []
    # for YEAR in YEAR_LST:
    for SEASON in SEASON_LST:
        fldr_lst.append(f"{INIT_PATH}/GSW/{SEASON}/")

    params = ((fldr, 'GSW') for fldr in fldr_lst)

    # run HUC12 zonal stats in 4 processes
    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=4
    # ) as executor:
    #     executor.map(param_huc12_wrapper, params)

    # run NWI zonal stats in 4 processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4
    ) as executor:
        executor.map(param_nwi_wrapper, params)

    return


if __name__ == '__main__':
    main()
