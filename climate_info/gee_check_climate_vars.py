### Must use an env with ee
# ex: global_flood_training
import pandas as pd
import altair as alt
import numpy as np
import folium

import ee
ee.Initialize()


def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=1000,
                                  crs='EPSG:4326',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):
  """Creates a region reduction function.

  Creates a region reduction function intended to be used as the input function
  to ee.ImageCollection.map() for reducing pixels intersecting a provided region
  to a statistic for each image in a collection. See ee.Image.reduceRegion()
  documentation for more details.

  Args:
    geometry:
      An ee.Geometry that defines the region over which to reduce data.
    reducer:
      Optional; An ee.Reducer that defines the reduction method.
    scale:
      Optional; A number that defines the nominal scale in meters of the
      projection to work in.
    crs:
      Optional; An ee.Projection or EPSG string ('EPSG:5070') that defines
      the projection to work in.
    bestEffort:
      Optional; A Boolean indicator for whether to use a larger scale if the
      geometry contains too many pixels at the given scale for the operation
      to succeed.
    maxPixels:
      Optional; A number specifying the maximum number of pixels to reduce.
    tileScale:
      Optional; A number representing the scaling factor used to reduce
      aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable
      computations that run out of memory with the default.

  Returns:
    A function that accepts an ee.Image and reduces it by region, according to
    the provided arguments.
  """

  def reduce_region_function(img):
    """Applies the ee.Image.reduceRegion() method.

    Args:
      img:
        An ee.Image to reduce to a statistic by region.

    Returns:
      An ee.Feature that contains properties representing the image region
      reduction results per band and the image timestamp formatted as
      milliseconds from Unix epoch (included to enable time series plotting).
    """

    stat = img.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        crs=crs,
        bestEffort=bestEffort,
        maxPixels=maxPixels,
        tileScale=tileScale)

    return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
  return reduce_region_function



# Define a function to transfer feature properties to a dictionary.
def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)


# Function to add date variables to DataFrame.
def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df



today = ee.Date(pd.to_datetime('2021'))
date_range = ee.DateRange(today.advance(-5, 'years'), today)
pdsi = ee.ImageCollection('GRIDMET/DROUGHT').filterDate(date_range).select('pdsi')
precip = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(date_range).select('pr')
tmmx = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(date_range).select('tmmx')


##### Upper Neuse #####
neuse_aoi = ee.FeatureCollection('users/mdgaines/HUC8_03020201_UpperNeuse').geometry()

### PDSI ###
neuse_reduce_pdsi = create_reduce_region_function(
    geometry=neuse_aoi, reducer=ee.Reducer.mean(), scale=5000, crs='EPSG:5070')

neuse_pdsi_stat_fc = ee.FeatureCollection(pdsi.map(neuse_reduce_pdsi)).filter(
    ee.Filter.notNull(pdsi.first().bandNames()))

neuse_pdsi_dict = fc_to_dict(neuse_pdsi_stat_fc).getInfo()

print(type(neuse_pdsi_dict), '\n')
for prop in neuse_pdsi_dict.keys():
    print(prop + ':', neuse_pdsi_dict[prop][0:3] + ['...'])

neuse_pdsi_df = pd.DataFrame(neuse_pdsi_dict)

neuse_pdsi_df = add_date_info(neuse_pdsi_df)
neuse_pdsi_df.head(5)

neuse_pdsi_df = neuse_pdsi_df.rename(columns={
    'pdsi': 'PDSI'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Neuse PDSI")
print(f"min: {round(neuse_pdsi_df['PDSI'].min(), 3)}")
print(f"median: {round(neuse_pdsi_df['PDSI'].median(), 3)}")
print(f"mean: {round(neuse_pdsi_df['PDSI'].mean(), 3)}")
print(f"max: {round(neuse_pdsi_df['PDSI'].max(), 3)}")


### Precip ###
reduce_precip = create_reduce_region_function(
    geometry=neuse_aoi, reducer=ee.Reducer.sum(), scale=4000, crs='EPSG:5070')

precip_stat_fc = ee.FeatureCollection(precip.map(reduce_precip)).filter(
    ee.Filter.notNull(precip.first().bandNames()))

precip_dict = fc_to_dict(precip_stat_fc).getInfo()

print(type(precip_dict), '\n')
for prop in precip_dict.keys():
    print(prop + ':', precip_dict[prop][0:3] + ['...'])

neuse_precip_df = pd.DataFrame(precip_dict)

neuse_precip_df = add_date_info(neuse_precip_df)
neuse_precip_df.head(5)

neuse_precip_df = neuse_precip_df.rename(columns={
    'pr': 'Precip'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Neuse Precip")
print(f"min: {round(neuse_precip_df['Precip'].min(), 3)}")
print(f"median: {round(neuse_precip_df['Precip'].median(), 3)}")
print(f"mean: {round(neuse_precip_df['Precip'].mean(), 3)}")
print(f"max: {round(neuse_precip_df['Precip'].max(), 3)}")


### Max TEMP ###
neuse_reduce_tmmx = create_reduce_region_function(
    geometry=neuse_aoi, reducer=ee.Reducer.mean(), scale=4000, crs='EPSG:5070')

neuse_tmmx_stat_fc = ee.FeatureCollection(tmmx.map(neuse_reduce_tmmx)).filter(
    ee.Filter.notNull(tmmx.first().bandNames()))

neuse_tmmx_dict = fc_to_dict(neuse_tmmx_stat_fc).getInfo()

print(type(neuse_tmmx_dict), '\n')
for prop in neuse_tmmx_dict.keys():
    print(prop + ':', neuse_tmmx_dict[prop][0:3] + ['...'])

neuse_tmmx_df = pd.DataFrame(neuse_tmmx_dict)

neuse_tmmx_df = add_date_info(neuse_tmmx_df)
neuse_tmmx_df.head(5)

neuse_tmmx_df = neuse_tmmx_df.rename(columns={
    'tmmx': 'MaxTemp'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Neuse MaxTemp")
print(f"min: {round(neuse_tmmx_df['MaxTemp'].min(), 3)}")
print(f"median: {round(neuse_tmmx_df['MaxTemp'].median(), 3)}")
print(f"mean: {round(neuse_tmmx_df['MaxTemp'].mean(), 3)}")
print(f"max: {round(neuse_tmmx_df['MaxTemp'].max(), 3)}")


alt.Chart(neuse_pdsi_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'mean(PDSI):Q', scale=alt.Scale(scheme='redblue', domain=(-5, 5))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('mean(PDSI):Q', title='PDSI')
    ]).properties(width=600, height=300)

alt.Chart(neuse_precip_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'sum(Precip):Q', scale=alt.Scale(scheme='bluepurple', domain=(0, 67000))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('sum(Precip):Q', title='Precip')
    ]).properties(width=600, height=300)

alt.Chart(neuse_tmmx_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'mean(MaxTemp):Q', scale=alt.Scale(scheme='yelloworangered', domain=(285, 305))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('mean(MaxTemp):Q', title='MaxTemp')
    ]).properties(width=600, height=300)



alt.Chart(neuse_pdsi_df).mark_bar(size=1).encode(
    x='Timestamp:T',
    y='PDSI:Q',
    color=alt.Color(
        'PDSI:Q', scale=alt.Scale(scheme='redblue', domain=(-5, 5))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('PDSI:Q', title='PDSI')
    ]).properties(width=600, height=300)

alt.Chart(neuse_precip_df).mark_bar(size=1).encode(
    x='Timestamp:T',
    y='Precip:Q',
    color=alt.Color(
        'Precip:Q', scale=alt.Scale(scheme='bluepurple', domain=(0, 30000))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('Precip:Q', title='Precip')
    ]).properties(width=600, height=300)

alt.Chart(neuse_tmmx_df).mark_bar(size=1).encode(
    x='Timestamp:T',
    y='MaxTemp:Q',
    color=alt.Color(
        'MaxTemp:Q', scale=alt.Scale(scheme='yelloworangered', domain=(285, 305))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('MaxTemp:Q', title='MaxTemp')
    ]).properties(width=600, height=300)




##### Upper Chattahoochee #####
chatt_aoi = ee.FeatureCollection('users/mdgaines/HUC8_03130001_UpperChat').geometry()

### PDSI ###
today = ee.Date(pd.to_datetime('2021'))
date_range = ee.DateRange(today.advance(-5, 'years'), today)
pdsi = ee.ImageCollection('GRIDMET/DROUGHT').filterDate(date_range).select('pdsi')

chatt_reduce_pdsi = create_reduce_region_function(
    geometry=chatt_aoi, reducer=ee.Reducer.mean(), scale=5000, crs='EPSG:5070')

chatt_pdsi_stat_fc = ee.FeatureCollection(pdsi.map(chatt_reduce_pdsi)).filter(
    ee.Filter.notNull(pdsi.first().bandNames()))

chatt_pdsi_dict = fc_to_dict(chatt_pdsi_stat_fc).getInfo()

print(type(chatt_pdsi_dict), '\n')
for prop in chatt_pdsi_dict.keys():
    print(prop + ':', chatt_pdsi_dict[prop][0:3] + ['...'])

chatt_pdsi_df = pd.DataFrame(chatt_pdsi_dict)

chatt_pdsi_df = add_date_info(chatt_pdsi_df)
chatt_pdsi_df.head(5)

chatt_pdsi_df = chatt_pdsi_df.rename(columns={
    'pdsi': 'PDSI'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Chatt PDSI")
print(f"min: {round(chatt_pdsi_df['PDSI'].min(), 3)}")
print(f"median: {round(chatt_pdsi_df['PDSI'].median(), 3)}")
print(f"mean: {round(chatt_pdsi_df['PDSI'].mean(), 3)}")
print(f"max: {round(chatt_pdsi_df['PDSI'].max(), 3)}")


### Precip ###
reduce_precip = create_reduce_region_function(
    geometry=chatt_aoi, reducer=ee.Reducer.sum(), scale=4000, crs='EPSG:5070')

precip_stat_fc = ee.FeatureCollection(precip.map(reduce_precip)).filter(
    ee.Filter.notNull(precip.first().bandNames()))

precip_dict = fc_to_dict(precip_stat_fc).getInfo()

print(type(precip_dict), '\n')
for prop in precip_dict.keys():
    print(prop + ':', precip_dict[prop][0:3] + ['...'])

chatt_precip_df = pd.DataFrame(precip_dict)

chatt_precip_df = add_date_info(chatt_precip_df)
chatt_precip_df.head(5)

chatt_precip_df = chatt_precip_df.rename(columns={
    'pr': 'Precip'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Chatt Precip")
print(f"min: {round(chatt_precip_df['Precip'].min(), 3)}")
print(f"median: {round(chatt_precip_df['Precip'].median(), 3)}")
print(f"mean: {round(chatt_precip_df['Precip'].mean(), 3)}")
print(f"max: {round(chatt_precip_df['Precip'].max(), 3)}")


### Max TEMP ###
chatt_reduce_tmmx = create_reduce_region_function(
    geometry=chatt_aoi, reducer=ee.Reducer.mean(), scale=4000, crs='EPSG:5070')

chatt_tmmx_stat_fc = ee.FeatureCollection(tmmx.map(chatt_reduce_tmmx)).filter(
    ee.Filter.notNull(tmmx.first().bandNames()))

chatt_tmmx_dict = fc_to_dict(chatt_tmmx_stat_fc).getInfo()

print(type(chatt_tmmx_dict), '\n')
for prop in chatt_tmmx_dict.keys():
    print(prop + ':', chatt_tmmx_dict[prop][0:3] + ['...'])

chatt_tmmx_df = pd.DataFrame(chatt_tmmx_dict)

chatt_tmmx_df = add_date_info(chatt_tmmx_df)
chatt_tmmx_df.head(5)

chatt_tmmx_df = chatt_tmmx_df.rename(columns={
    'tmmx': 'MaxTemp'
}).drop(columns=['millis', 'system:index'])
# pdsi_df.head(5)

print("Chatt MaxTemp")
print(f"min: {round(chatt_tmmx_df['MaxTemp'].min(), 3)}")
print(f"median: {round(chatt_tmmx_df['MaxTemp'].median(), 3)}")
print(f"mean: {round(chatt_tmmx_df['MaxTemp'].mean(), 3)}")
print(f"max: {round(chatt_tmmx_df['MaxTemp'].max(), 3)}")



def get_seasonal_avg_pdsi(pdsi_df, year):
   spring_pdsi = pdsi_df.loc[((pdsi_df['Month']>=3) & (pdsi_df['Month']<6)), 'PDSI'].mean()
   summer_pdsi = pdsi_df.loc[((pdsi_df['Month']>=6) & (pdsi_df['Month']<9)), 'PDSI'].mean()
   fall_pdsi = pdsi_df.loc[((pdsi_df['Month']>=9) & (pdsi_df['Month']<12)), 'PDSI'].mean()
   winter_pdsi = pdsi_df.loc[((pdsi_df['Month']>=12) | (pdsi_df['Month']<3)), 'PDSI'].mean()

   print(f"{year}\n\tspring:\t{spring_pdsi}\n\tsummer:\t{summer_pdsi}\n\tfall:\t{fall_pdsi}\n\twinter:\t{winter_pdsi}")


chatt_pdsi_2017_df = chatt_pdsi_df[((chatt_pdsi_df['Year']==2017) & (chatt_pdsi_df['Month']>=3)) | 
                                   (chatt_pdsi_df['Year']==2018) & (chatt_pdsi_df['Month']<3)]
get_seasonal_avg_pdsi(chatt_pdsi_2017_df, 2017)
'''
2017
	spring:	-4.705940025799912
	summer:	-2.4294385782503665
	fall:	-0.60439063274147
	winter:	-1.0356908408049983
'''
chatt_pdsi_2019_df = chatt_pdsi_df[((chatt_pdsi_df['Year']==2019) & (chatt_pdsi_df['Month']>=3)) | 
                                   (chatt_pdsi_df['Year']==2020) & (chatt_pdsi_df['Month']<3)]
get_seasonal_avg_pdsi(chatt_pdsi_2019_df, 2019)
'''
2019
	spring:	4.9656755308461245
	summer:	3.8600105427765197
	fall:	1.7894875035476723
	winter:	1.4767297220687015
'''

alt.Chart(chatt_pdsi_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'mean(PDSI):Q', scale=alt.Scale(scheme='redblue', domain=(-5, 5))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('mean(PDSI):Q', title='PDSI')
    ]).properties(width=600, height=500).configure_axis(
       titleFontSize=24, labelFontSize=20).configure_scale(minFontSize=24)

alt.Chart(chatt_precip_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'sum(Precip):Q', scale=alt.Scale(scheme='bluepurple', domain=(0, 67000))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('sum(Precip):Q', title='Precip')
    ]).properties(width=1200, height=600)

alt.Chart(chatt_tmmx_df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'mean(MaxTemp):Q', scale=alt.Scale(scheme='yelloworangered', domain=(285, 305))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('mean(MaxTemp):Q', title='MaxTemp')
    ]).properties(width=600, height=300)



alt.Chart(chatt_pdsi_df).mark_bar(size=3).encode(
    x='Timestamp:T',
    y='PDSI:Q',
    color=alt.Color(
        'PDSI:Q', scale=alt.Scale(scheme='redblue', domain=(-5, 5))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('PDSI:Q', title='PDSI')
    ]).properties(width=1000, height=500).configure_scale(
       minSize = 20, minFontSize=24).configure_axis(
       titleFontSize=24, labelFontSize=20)

alt.Chart(chatt_precip_df).mark_bar(size=1).encode(
    x='Timestamp:T',
    y='Precip:Q',
    color=alt.Color(
        'Precip:Q', scale=alt.Scale(scheme='bluepurple', domain=(0, 30000))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('Precip:Q', title='Precip')
    ]).properties(width=600, height=300)

alt.Chart(chatt_tmmx_df).mark_bar(size=1).encode(
    x='Timestamp:T',
    y='MaxTemp:Q',
    color=alt.Color(
        'MaxTemp:Q', scale=alt.Scale(scheme='yelloworangered', domain=(285, 305))),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Date'),
        alt.Tooltip('MaxTemp:Q', title='MaxTemp')
    ]).properties(width=600, height=300)
