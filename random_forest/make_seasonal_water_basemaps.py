import concurrent.futures
import json
import time
import numpy as np
from  rasterio.warp import transform_geom
from rasterio.mask import mask
import datetime as dt
import basemaps_internal
import os
from rasterio.plot import reshape_as_image
import joblib
from raster_functions import *
from download_data.download_basemaps import mk_basemap_dirs, get_mosaic_names_lst, get_year_season_sample_names
# from RF_train_test_save import str_class_to_int
from random_forest.compile_classified_points import print_time

def _calculate_bands(data, quad_name):

    start_bandMath = time.time()
    ## CALCULATE INDICIES ##
    ndwi_layer = ndwi(data)
    ndvi_layer = ndvi(data)
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
    
    end_bandMath = time.time()
    print_time(start_bandMath, end_bandMath, "\tband math")

    ## CALCULATE 3x3 WINDOW AVG ##
    start_windowCalc = time.time()
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
    
    end_windowCalc = time.time()
    print_time(start_windowCalc, end_windowCalc, "\twindow calc and stack")

    return stacked_data

def _run_rf(stacked_data, rf):
    
    start_rf = time.time()

    reshaped_stack = reshape_as_image(stacked_data.values)
    # print(reshaped_stack.shape, type(reshaped_stack))

    # subset to the 11 bands we are using
    index_bands_remove = [1, 2, 7, 8, 15]
    X = np.delete(reshaped_stack.T, index_bands_remove, axis=0).T
    # Bands used in RF:
    # 'blue', 'nir',
    # 'ndwi', 'ndvi',
    # 'blue_3x3', 'nir_3x3',
    # 'ndwi_3x3', 'ndvi_3x3',
    # 'elev', 'slope', 'hillshade'

    class_pred = rf.predict(X.reshape(-1, 11))
    # Reshape our classification map back into a 2D matrix so we can visualize it
    class_pred = class_pred.reshape(reshaped_stack[:, :, 0].shape)
    # class_pred.shape
    end_rf = time.time()
    print_time(start_rf, end_rf, "\tRF reshape, predict, reshape")

    return class_pred

def _save_class_raster(stacked_data, class_pred, outpath):
    # Add back spatial data
    x_coords = stacked_data.coords['x'].values
    y_coords = stacked_data.coords['y'].values
    spatial_coords = stacked_data.coords['spatial_ref']
    # Convert to xarray spatial DataArray
    class_xr = xr.DataArray(class_pred, 
                            coords={'x': x_coords, 'y': y_coords, 'spatial_ref': spatial_coords},
                            dims=['y', 'x'])
    # update band name and add No Data layer (alpha)
    class_rxr = xr.concat([class_xr.assign_coords({'band':'class'}), 
                        stacked_data[-1]],
                        dim='band').astype(int)
    # Save to raster (compressed)
    class_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW")
    del(class_rxr)

    return

def quadProcess(quad, quad_name, rf, out_dir):

    start_quad = time.time()
    # check we have not already
    # quad_name = quad_lst[0].id
    outpath = os.path.join(out_dir, quad_name + '_classified.tif')
    if os.path.exists(outpath):
        end_quad = time.time()
        print_time(start_quad, end_quad, f"{outpath} exists")
        return

    # quad = quad_lst[0]
    data = rxr.open_rasterio(quad.download_url)
    # band 1 : blue
    # band 2 : green
    # band 3 : red
    # band 4 : nir
    # band 5 : alpha (mask)
    # print(data.rio.crs)
    data = data.assign_coords({'band':['blue', 'green', 'red', 'nir', 'alpha']})

    #### CALCULATE BANDS ####
    stacked_data = _calculate_bands(data, quad_name)
    #### RUN RANDOM FOREST ####
    class_pred = _run_rf(stacked_data, rf)
    #### SAVE CLASSIFIED RASTER ####
    _save_class_raster(stacked_data, class_pred, outpath)

    end_quad = time.time()
    print_time(start_quad, end_quad, f"{outpath} saved")

    return

def param_wrapper(p):
    return quadProcess(*p)

def get_region():
    upper_chat_geojson_path = '../data/NWI/HU8_03130001_Watershed/HU8_03130001_Watershed.geojson'

    with open(upper_chat_geojson_path) as json_data:
        d = json.load(json_data)
    region = d['features'][0]['geometry']
    # print('Geometry to be used when querying for mosaics: {}'.format(region))
    return region

def main():

    start_main = time.time()

    # Authenticate PL_API_KEY
    # Set your api_key or have it stored as environment variable
    CLIENT = basemaps_internal.BasemapsClient(api_key=os.environ.get('ACCESS_KEY_ID'))

    # INIT_PATH = '../data/PlanetBasemaps'
    YEAR_PATHS = ['2017', '2019']
    SEASON_PATHS = ['SPRING', 'SUMMER', 'FALL', 'WINTER']
    OUT_PATH = 'D:/Research/data/PlanetBasemaps'

    # import random forest
    # rf = joblib.load('D:/Research/data/RF_compressed.joblib')
    rf = joblib.load('D:/Research/data/RF_SRTM_AUC_compressed.joblib')

    # import AOI shapefile
    region = get_region()

    series_name = 'PS normalized_analytic weekly subscription'
    series = CLIENT.series(name=series_name)

    for year in [2017, 2019]:
        start_year = time.time()

        print(f'\n\nYEAR: {year}')

        start = dt.datetime(year, 2, 24) 
        end = dt.datetime(year+1, 3, 5)
        # get list of mosaic names
        mosaic_name_lst = get_mosaic_names_lst(series = series, 
                                            start = start, 
                                            end = end,
                                            region = region)

        szn_mosaic_name_lsts = get_year_season_sample_names(mosaic_name_lst, year, seed=0, samp=0)
        for i in range(len(szn_mosaic_name_lsts)):
            start_szn = time.time()
            # i = 0
            szn = SEASON_PATHS[i]
            print(f"\tSeason: {szn}")
            for mosaic_name in szn_mosaic_name_lsts[i]:
                start_mosaic = time.time()

                mosaic = CLIENT.mosaic(name=mosaic_name)
                mosaic_name_short = '_'.join(mosaic_name.split('_')[-3:]).replace('-','')
                out_dir = os.path.join(OUT_PATH, str(year), szn, mosaic_name_short)
                mk_basemap_dirs(out_dir)

                quad_lst = list(mosaic.quads(region=region))

                params = ((quad, quad.id, rf, out_dir) for quad in quad_lst)
                # can use ProcessPoolExecutor because we are reading and writing different files in each core
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=7
                ) as executor:
                    executor.map(param_wrapper, params)
                
                end_mosaic = time.time()
                print_time(start_mosaic, end_mosaic, mosaic_name_short)
                # return # breaking the loops for testing
            
            end_szn = time.time()
            print_time(start_szn, end_szn, szn)

        end_year = time.time()
        print_time(start_year, end_year, f"{year}")

    end_main = time.time()
    print_time(start_main, end_main, "full run of main")


if __name__ == '__main__':
    main()

