import json
import pprint
import shapely
import numpy as np
import geopandas as gpd
import rasterio as rio
from  rasterio.warp import transform_geom
from rasterio.mask import mask
import datetime as dt
import basemaps_internal
import random
import os

# Authenticate PL_API_KEY
# Set your api_key or have it stored as environment variable
client = basemaps_internal.BasemapsClient(api_key=os.environ.get('ACCESS_KEY_ID'))

INIT_PATH = '../../data/PlanetBasemaps'
YEAR_PATHS = ['2017', '2019']
SEASON_PATHS = ['SPRING', 'SUMMER', 'FALL', 'WINTER']

def mk_basemap_dirs(dir):

    if not os.path.exists(dir):
        os.makedirs(dir)

    # for i in range(len(YEAR_PATHS)):
    #     for j in range(len(SEASON_PATHS)):
    #         if not os.path.exists(os.path.join(INIT_PATH, YEAR_PATHS[i], SEASON_PATHS[j])):
    #             os.makedirs(os.path.join(INIT_PATH, YEAR_PATHS[i], SEASON_PATHS[j]))
    return

def get_mosaic_names_lst(series, start, end, region, mosaic_name_lst:list=[]):
    # Query each mosaic to print out info about how much data we'll download
    # Select the only (first) geometry in region
    for mosaic in series.mosaics(start_date=start, end_date=end):
        nquads = len(list(mosaic.quads(region=region)))
        print('For {}, there are {} quads available!'.format(mosaic.name, nquads))
        mosaic_name_lst.append(mosaic.name)
    return mosaic_name_lst

def get_year_season_sample_names(mosaic_name_lst:list, yr:int, seed:int, samp:int=6):
    # set random seed
    if samp == 6 or samp == 0:
        separator = '-'
    else: 
        separator = ''
    random.seed(seed)

    spring_lst = [i for i in mosaic_name_lst if \
                    ((f'{yr}{separator}03{separator}' in i) 
                    or (f'{yr}{separator}04{separator}' in i) 
                    or (f'{yr}{separator}05{separator}' in i)) \
                        and not (f'{yr}{separator}06{separator}' in i) ]

    summer_lst = [i for i in mosaic_name_lst if \
                    ((f'{yr}{separator}06{separator}' in i) 
                    or (f'{yr}{separator}07{separator}' in i) 
                    or (f'{yr}{separator}08{separator}' in i)) \
                        and not (f'{yr}{separator}09{separator}' in i) ]

    fall_lst = [i for i in mosaic_name_lst if \
                    ((f'{yr}{separator}09{separator}' in i) 
                    or (f'{yr}{separator}10{separator}' in i) 
                    or (f'{yr}{separator}11{separator}' in i)) \
                        and not (f'{yr}{separator}12{separator}' in i) ]

    winter_lst = [i for i in mosaic_name_lst if  \
                    ((f'{yr}{separator}12{separator}' in i) 
                    or (f'{yr+1}{separator}01{separator}' in i) 
                    or (f'{yr+1}{separator}02{separator}' in i)) \
                    and not (f'{yr+1}{separator}03{separator}' in i) ]

    if samp == 0:
        return [spring_lst, summer_lst, fall_lst, winter_lst]


    spring_samp_lst = random.sample(spring_lst, samp)
    summer_samp_lst = random.sample(summer_lst, samp)
    fall_samp_lst = random.sample(fall_lst, samp)
    winter_samp_lst = random.sample(winter_lst, samp)

    return [spring_samp_lst, summer_samp_lst, fall_samp_lst, winter_samp_lst]


def download_basemap(client, name, region, out_dir, seed:int, quad_id_lst:list,
                     quad_dict:dict):
    # set random seed
    random.seed(seed)

    # Downloading a single quad for a SR normalized mosaic
    mosaic = client.mosaic(name=name)

    quad_lst = list(mosaic.quads(region=region))

    quad_lst2 = [quad for quad in quad_lst if (quad.coverage > 50) and (quad.id in quad_id_lst) ]
    print(f'{name} has {len(quad_lst2)} >50% useful quads')

    if len(quad_lst2) > 10:
        quad_samp_lst = random.sample(quad_lst2, 10)
    else:
        quad_samp_lst = quad_lst2

    for quad in quad_samp_lst:
        # Download quad locally
        # print(quad_lst.index(quad))
        # print(quad.items.id)
        # print(quad.items.percent_covered)
        loc = quad.download(output_dir=out_dir)
        print('Quad was downloaded locally, and saved at: {}'.format(loc))

        quad_dict[loc] = {'mosaic': name, 'quad_id': quad.id, 'coverage': quad.coverage}
    
    return quad_dict


def save_dict(dct):
    # create json object from dictionary
    jsn = json.dumps(dct)

    # open file for writing, "w" 
    f = open(f"{INIT_PATH}/quad_dict.json","w")

    # write json object to file
    f.write(jsn)

    # close file
    f.close()

    return



def main():

    # import AOI shapefile
    upper_chat_geojson_path = '../../data/NWI/HU8_03130001_Watershed/HU8_03130001_Watershed.geojson'
    if not os.path.exists(upper_chat_geojson_path):
        upper_chat_shp = gpd.read_file('../../data/NWI/HU8_03130001_Watershed/HU8_03130001_Watershed.shp')
        # reproject
        upper_chat_shp = upper_chat_shp.to_crs('EPSG:4326')
        # save as geojson
        upper_chat_shp.geometry.to_json()
        upper_chat_shp.geometry.to_file(upper_chat_geojson_path, driver='GeoJSON')

    with open(upper_chat_geojson_path) as json_data:
        d = json.load(json_data)
        # print(d)
    region = d['features'][0]['geometry']
    # print('Geometry to be used when querying for mosaics: {}'.format(region))

    # import quad exlusion list
    quads_shp = gpd.read_file('../../data/PlanetBasemaps/quads_percentOverlap.shp')
    quad_id_lst = list(quads_shp[quads_shp['HU8_0313_1'] > 30]['quad_id'])
 

    # get list of mosaic names
    quad_dict = {}
    series_name = 'PS normalized_analytic weekly subscription'
    series = client.series(name=series_name)

    for yr in [17, 19]:
        print(f'\n\nYEAR: {yr}')
    
        start = dt.datetime(2000+yr, 2, 24) 
        end = dt.datetime(2000+yr+1, 3, 5)

        mosaic_name_lst = get_mosaic_names_lst(series = series, 
                                            start = start, 
                                            end = end,
                                            region = region)
        
        season_samp_lst = get_year_season_sample_names(mosaic_name_lst, yr=yr, seed=yr)

        for i in range(4):
            samp_lst = season_samp_lst[i]
            year = f'20{yr}'
            szn = SEASON_PATHS[i]
            print(f'\nseason: {szn}')
            for j in range(len(samp_lst)):
                mosaic_name = samp_lst[j]
                mosaic_name_short = '_'.join(mosaic_name.split('_')[-3:]).replace('-','')
                out_dir = os.path.join(INIT_PATH, year, szn, mosaic_name_short)
                mk_basemap_dirs(out_dir)
                # set random seed based on mosaic_name index
                seed = mosaic_name_lst.index(mosaic_name)
                print(f'\nseed: {seed}')
                quad_dict = download_basemap(client, name=mosaic_name, region=region, 
                                out_dir=out_dir, seed=seed, quad_id_lst=quad_id_lst, 
                                quad_dict=quad_dict)

    save_dict(quad_dict)


if __name__ == '__main__':
    main()