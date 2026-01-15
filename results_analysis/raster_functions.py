from glob import glob
import numpy as np
import rioxarray as rxr
import xarray as xr
from itertools import product

from osgeo import gdal
import os

INIT_PATH = 'D:/Research'

def stack_bands(full_rstr:xr.core.dataarray.DataArray, band_lst:list, window:bool):
    '''
    Returns stacked xarray DataArray with bands named
        full_rstr: full raster as xarray DataArray
        band_lst: list of bands to add to full_rstr
        window: True/False if stacking 3x3 window average bands
    '''
    # If stacking 3x3 window average bands
    if window:
        # Set alpha (no data) index
        alpha_idx = 6
        # List of 3x3 average bands
        band_name_lst_3x3 = ['blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3',
                         'ndwi_3x3', 'ndvi_3x3']
        # List of SRTM DEM bands
        band_name_lst_srtm = ['elev', 'slope', 'hillshade']
        # Stack all 3x3 average bands into an xr.DataArray
        index_stack_3x3 = xr.concat(band_lst[0:6], dim='band')
        # Stack all SRTM DEM bands into an xr.DataArray
        index_stack_srtm = xr.concat(band_lst[6:], dim='band')
        # Stack the 3x3 average and SRTM DEM bands together
        index_stack = xr.concat([index_stack_3x3.assign_coords({'band': band_name_lst_3x3}),
                                index_stack_srtm], dim='band')
        # Update the band name list of the stacked DataArray
        band_name_lst = ['blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3',
                         'ndwi_3x3', 'ndvi_3x3',
                         'elev', 'slope', 'hillshade']

    # If stacking non moving window bands
    else:
        # Set alpha (no data) index
        alpha_idx = 4
        # List of bands to stack
        band_name_lst = ['ndwi', 'ndvi']
        # Stack bands
        index_stack = xr.concat(band_lst, dim='band')
    
    # Combine initial raster (ex bands: BLUE, GREEN, RED, NIR) with the newly stacked bands
    stacked_data = xr.concat([
                        full_rstr[0:alpha_idx],
                        index_stack.assign_coords({'band': band_name_lst}),
                        full_rstr[alpha_idx]],
                    dim='band')
    
    return stacked_data


def ndwi(data:xr.core.dataarray.DataArray):
    '''
        Calculate NDWI from Planet Basemap quad (DataArray)
        NDWI = (GREEN - NIR) / (GREEN + NIR)
    '''
    return (data[1].astype(float) - data[3]) / (data[1] + data[3])

def ndvi(data:xr.core.dataarray.DataArray):
    '''
        Calculate NDVI from Planet Basemap quad (DataArray)
        NDVI = (NIR - RED) / (NIR + RED)
    '''
    return (data[3].astype(float) - data[2]) / (data[3] + data[2])


def check_resamp_exists(quad_name:str, data_name:str):
    '''
        Check if the resampled raster has been calculated and saved
    '''
    # build resampled raster path
    resamp_tif_path = os.path.join(INIT_PATH, 'data', 'DEMs', data_name, f'{data_name}_quad_{quad_name}.tif')
    # check if the resampled raster exists
    if os.path.exists(resamp_tif_path):
        return 'y', resamp_tif_path
    else:
        return 'n', resamp_tif_path

def resamp(srtm_rxr:xr.core.dataarray.DataArray, data_rxr:xr.core.dataarray.DataArray, outpath:str):
    '''
        Reproject and resample the SRTM data (elevation, slope, or hillshade) to match the
        Planet Basemap data
    '''
    # reproject, resample
    resamp_rxr1 = srtm_rxr.rio.reproject_match(match_data_array=data_rxr, nodata=255)
    resamp_rxr = resamp_rxr1.assign_coords({'x':data_rxr.x,
                                            'y':data_rxr.y})
    # save reprojected/resampled raster
    resamp_rxr = resamp_rxr.rio.write_nodata(255, inplace=True)
    resamp_rxr.rio.to_raster(outpath, driver="GTiff", compress="LZW", dtype="uint8")
    print(f'{os.path.basename(outpath)} saved.')
    # return the DataArray
    return resamp_rxr

def get_srtm_data(data_rxr:xr.core.dataarray.DataArray, quad_name:str, data_name:str):
    '''
        Get the SRTM data (elevation, slope, or hillshade) that matches the 
        Plane Basemap data
    '''
    # check if the data already exists in the correct projection
    y_n, resamp_tif_path = check_resamp_exists(quad_name, data_name)
    # if not, resampled the original raster and save it for future use
    if y_n == 'n':
        # path to the original raster
        tif_path = os.path.join(INIT_PATH, 'data', 'DEMs', f'nGA_{data_name}.tif')
        # open the original raster
        srtm_data = rxr.open_rasterio(tif_path, lock=True)
        # resample and save
        resamp_srtm = resamp(srtm_data, data_rxr, resamp_tif_path)
    else:
        # open the resampled raster
        resamp_srtm1 = rxr.open_rasterio(resamp_tif_path, lock=True)
        resamp_srtm = resamp_srtm1.assign_coords({'x':data_rxr.x,
                                                  'y':data_rxr.y})
    # return the resampled raster
    return resamp_srtm[0]


def moving_window_avg(raster:xr.core.dataarray.DataArray, window:int):
    '''
        Modified from https://pygis.io/docs/e_raster_window_operations.html
        
        Calculates the local average for each pixel within a moving window
    '''
    # Create a kernel to calculate the average
    weight = 1 / (window ** 2)
    kernel = np.full((window, window), weight)
    # Get kernel shape
    kernel_shape = kernel.shape
    # Convert the kernel to a flattened array
    kernel_array = np.ravel(kernel)
    
    # Create raster array with placeholder values in shape of raster
    output_rio = np.full((raster.shape[0], raster.shape[1]), -9999) # originally -9999
    # Set array data type
    output_rio = output_rio.astype(np.float32) # originally np.float64
    
    # Create raster array used to store window operation calculations for each pixel (excluding boundary pixels)
    aggregate = np.full((raster.shape[0] - kernel_shape[0] + 1, raster.shape[1] - kernel_shape[1] + 1), 0)
    # Set array data type
    aggregate = aggregate.astype(np.float32) # originally np.float64
    
    # Generate row index pairs for slicing
    pairs_x = list(zip([None] + list(np.arange(1, kernel_shape[0])), list(np.arange(-kernel_shape[0] + 1, 0)) + [None]))
    # Generate column index pairs for slicing
    pairs_y = list(zip([None] + list(np.arange(1, kernel_shape[1])), list(np.arange(-kernel_shape[1] + 1, 0)) + [None]))
    # Combine row and column index pairs together to get the extent for each vectorized sliding window
    combos = list(product(pairs_x, pairs_y))
    
    # Iterate through the combined pairs (which give extent of a sliding window)
    for p in range(len(combos)):
        # Get the sub-array via slicing and multiply all the values by corresponding value in kernel (based on location)
        sub_array = raster[combos[p][0][0]:combos[p][0][1], combos[p][1][0]:combos[p][1][1]] * kernel_array[p]
        # Add sub-array values to array storing window operation calculations
        aggregate += sub_array
    
    # Use kernel shape to determine the row and column index extent of the calculated array
    n = int((kernel_shape[0] - 1) / 2)
    m = int((kernel_shape[1] - 1) / 2)

    # Replace placeholder values in the output array with the corresponding values (based on location) from the calculated array
    output_rio[n:-n, m:-m] = aggregate
      
    return output_rio

def moving_window_band_loop(full_rstr:xr.core.dataarray.DataArray, window:int):
    '''
    Returns a list of bands (xarray DataArray) that are the window average 
        of each of the four Planet bands (blue, green, red, nir)

        full_rstr: xarray DataArray of the Planet Normalized Basemap
                   no data === -9999
                   bands: blue, green, red, nir, alpha
    '''
    # calculate 3x3 moving window average for each band (excluding alpha)
    x_coords = full_rstr.coords['x'].values
    y_coords = full_rstr.coords['y'].values
    spatial_coords = full_rstr.coords['spatial_ref']

    window_lst = [] # empty list for each 3x3 band
    for band_idx in range(len(full_rstr.band)-1):
        band = full_rstr[band_idx].values
        band_name = full_rstr.band.values[band_idx]
        # calculate moving window
        window_avg_band = moving_window_avg(band, window)
        window_lst.append(
            # convert to xarray DataArray with spatial data
            xr.DataArray(window_avg_band,
                        coords={'x': x_coords, 'y': y_coords, 'spatial_ref': spatial_coords},
                        dims=['y', 'x']))
        
    return window_lst


def clean_window_avg(rstr:xr.core.dataarray.DataArray):
    '''
    Returns stacked raster with corrected 3x3 averages
        replace values that used the -9999 in the avg with the original band value
        rstr: full stacked raster as xarray DataArray
    '''
    for band_3x3 in ['blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3',
                'ndwi_3x3', 'ndvi_3x3']:
        band = band_3x3[:-4]
        if band in ['ndwi', 'ndvi']:
            # threshold of -10 because ndvi and ndwi can range from -1 to 1
            # -10 just in case something weird happens
            threshold = -10
        else:
            # threshold of 0 because non-indices bands should be positive
            threshold = 0
        # Replace values that are below the threshold with the original band value
        # ( xr.DataArray.where() preserves values where the condition (band > threshold) is TRUE
        #   and fills in data where the condition (band > threshold) is FALSE)
        rstr.loc[band_3x3] = rstr.loc[band_3x3].where(
            rstr.loc[band_3x3] > threshold, rstr.loc[band])
        
    return rstr

