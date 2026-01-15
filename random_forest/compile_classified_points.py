import concurrent.futures
import os
from glob import glob
import numpy as np
import geopandas as gpd
import time
import random
import sys

import rioxarray as rxr
from rasterio.plot import plotting_extent

import rasterstats as rs
from raster_functions import *
from download_data.download_basemaps import get_year_season_sample_names

def print_time(start, end, process = ""):
    """
        prints message with elapsed time of process
    """
    elapsed = end - start
    time_str = time.strftime("%H hr %M min %S sec", time.gmtime(elapsed))
    print(f"[{os.getpid()}]\t{process} completed in {time_str}", flush=True)
    return

def get_point_raster_vals(shp_path:str):
    '''
    Saves shapefiles of classified points with all 13 raster band values 
        to be used for random forest training
    '''
    start_time = time.time()
    outpath = f'../../data/PlanetBasemaps/mosaic_shapefiles/{os.path.basename(shp_path)[0:24]}.shp'

    if os.path.exists(outpath):
        end_time = time.time()
        print_time(start_time, end_time, f'{os.path.basename(outpath)} exists')
        return

    # import classified points
    shp = gpd.read_file(shp_path)
    bands = ['blue', 'green', 'red', 'nir', 
            'ndwi', 'ndvi', 
            'blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3', 
            'ndwi_3x3', 'ndvi_3x3',
            'elev', 'slope', 'hillshade', 
            'alpha']
    for band in bands:
        shp.loc[:,band] = -9999.
        

    mosaic_path = os.path.dirname(shp_path)
    rstr_lst = glob(mosaic_path + '/*.tif')
    quad_count = 0
    quad_total = len(rstr_lst)
    for rstr_path in rstr_lst:
        start_quad = time.time()
        # import raster
        data = rxr.open_rasterio(rstr_path)
        # band 1 : blue
        # band 2 : green
        # band 3 : red
        # band 4 : nir
        # band 5 : alpha (mask)
        # print(data.rio.crs)
        data = data.assign_coords({'band':['blue', 'green', 'red', 'nir', 'alpha']})

        #### CALCULATE BANDS ####

        ## CALCULATE INDICIES ##
        ndwi_layer = ndwi(data)
        ndvi_layer = ndvi(data)
        quad_name = os.path.basename(rstr_path)[:-4]
        elev = get_srtm_data(data_rxr=data, quad_name=quad_name, data_name='elev')
        slope = get_srtm_data(data_rxr=data, quad_name=quad_name, data_name='slope')
        hillshade = get_srtm_data(data_rxr=data, quad_name=quad_name, data_name='hillshade')
        # stack bands 
        # band order:
        #             'blue', 'green', 'red', 'nir',
        #             'ndwi','ndvi'
        #             'alpha'
        stacked_data = stack_bands(data, [ndwi_layer, ndvi_layer], window=False)
        del(data)
        # mask unusable data (alpha band)
        data_clean = stacked_data.where(stacked_data[6] != 0, -9999)
        del(stacked_data)

        ## CALCULATE 3x3 WINDOW AVG ##
        window_lst = moving_window_band_loop(data_clean, window=3)
        window_lst.append(elev)
        window_lst.append(slope)
        window_lst.append(hillshade)
        # stack bands 
        # band order:
        #             'blue', 'green', 'red', 'nir',
        #             'ndwi', 'ndvi',
        #             'blue_3x3', 'green_3x3', 'red_3x3', 'nir',
        #             'ndwi_3x3', 'ndvi_3x3'
        #             'elev', 'slope', 'hillshade'
        #             'alpha'
        stacked_data = stack_bands(data_clean, window_lst, window=True)
        del(data_clean)
        del(window_lst)

        stacked_data = clean_window_avg(stacked_data)

        # mask unusable data (alpha band)
        stacked_data = stacked_data.where(stacked_data[-1] >= 0, -9999)

        if not stacked_data.shape == (16, 4096, 4096):
            print(f"Shape issue: {stacked_data.shape}")
            os._exit()

        #### EXTRACT POINT VALUES ####
        data_plotting_extent = plotting_extent(stacked_data[0], stacked_data.rio.transform())

        xmin = data_plotting_extent[0]
        xmax = data_plotting_extent[1]
        ymin = data_plotting_extent[2]
        ymax = data_plotting_extent[3]

        subset_shp = shp.cx[xmin:xmax, ymin:ymax]

        if subset_shp.crs != stacked_data.rio.crs:
            print('CRS issue')
            return

        for i in range(len(stacked_data)):
            band = bands[i]
            point_vals = rs.point_query(shp.cx[xmin:xmax, ymin:ymax], 
                                        stacked_data.values[i], 
                                        nodata=np.nan, 
                                        affine=stacked_data.rio.transform(), 
                                        interpolate='nearest')
            subset_shp.loc[:,band] = point_vals

        idx = shp.cx[xmin:xmax, ymin:ymax].index
        shp.iloc[idx] = subset_shp

        del(stacked_data)

        end_quad = time.time()
        quad_count += 1
        print_time(start_quad, end_quad, f'{quad_count}/{quad_total}')

    shp.to_file(outpath)

    end_time = time.time()
    print_time(start_time, end_time, f'{os.path.basename(outpath)} saved')

    return


def update_shapefiles():
    #### Get list of all mosaic shapefiles
    shp_lst = glob('../../data/PlanetBasemaps/20*/*/*/*points_*.shp')
    print(shp_lst)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1
        ) as executor:
        executor.map(get_point_raster_vals, shp_lst)

    return


def compile_shp(lst:list, outpath:str):
    start_compile = time.time()

    num_points = 0
    for i in range(len(lst)):
        shp_i = gpd.read_file(lst[i])
        num_points += len(shp_i)
        if i == 0:
            shp_full = shp_i
        else:
            if shp_full.crs != shp_i.crs:
                print('diff proj...how??')
                shp_i = shp_i.to_crs(shp_full.crs)
            shp_full = gpd.pd.concat([shp_full, shp_i])

    ### MAKE SURE ALL LABLES ARE CORRECT
    ## water, non-water, or null
    print(shp_full['class'].unique())
    shp_full.loc[shp_full['class']=='warer','class'] = 'water'
    shp_full.loc[shp_full['class']=='nul','class'] = 'null'

    try:
        shp_full.loc[shp_full['class_mdg']=='water', 'class'] = 'water'
        shp_full.loc[shp_full['class_mdg']=='non-water', 'class'] = 'non-water'
    except KeyError:
        print('No "mdg_class"')

    shp_full.to_file(f'D:/Research/data/Shapefiles/{outpath}')

    end_compile = time.time()
    print_time(start_compile, end_compile, f'compiled shps and saved {outpath}')


def main():
    update_shapefiles()

    #### Get list of all mosaic shapefiles
    shp_lst = glob('../../data/PlanetBasemaps/mosaic_shapefiles/*.shp')

    exclude_2017_lst = get_year_season_sample_names(shp_lst, yr=2017, seed=2017, samp=1)
    exclude_2017_lst = [i[0] for i in exclude_2017_lst]
    exclude_2019_lst = get_year_season_sample_names(shp_lst, yr=2019, seed=2019, samp=1)
    exclude_2019_lst = [i[0] for i in exclude_2019_lst]

    exclude_lst = exclude_2017_lst + exclude_2019_lst

    shp_lst = [i for i in shp_lst if i not in exclude_lst]

    compile_shp(lst=shp_lst, outpath='train_classified_points.shp')
    compile_shp(lst=exclude_lst, outpath='test_classified_points.shp')


    return

if __name__ == '__main__':
    main()

