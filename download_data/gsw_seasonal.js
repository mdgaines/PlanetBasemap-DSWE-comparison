// ===============================================
// GSW - Seasonal Composites
// ===============================================
//
// gsw_seasonal.js
//
// This script creates seasonal composites of Global Surface Water (GSW) v4 products: 
// categories of ground surface inundation as detected in cloud-/shadow-/snow-free 
// Landsat pixels. Where multiple values are returned for pixels within a month,
// water categories take precedence over non-water or cloudy pixels.
//
// GSW v4 data: 
// Seasonal GSW v4 coding: Mollie D. Gaines
//
// Citations: 
//
//  Pekel, J.F., Cottam, A., Gorelick, N., Belward, A.S., High-resolution mapping of global surface water and its long-term changes.
//                  Nature 540, 418-422 (2016). (doi:10.1038/nature20584)
//
// -------------------------------------------
// Input: User-supplied area of interest (aoi) and dates (within Landsat TM availability)
// Output: Multiple (8) images of seasonal GSW composites in Google Drive
// Output file name is automatically set to export with the name
// "GSW_<year>_<season>"
// but can be changed when starting the task in GEE.
//        
// GSW categories:
// 0 - No data
// 1 - Not water
// 2 - Water 
//
// Output categories:
// 0 - Not water
// 1 - Water
// ----------------------------------------------------------------------
var proj = ee.Projection('EPSG:4326')
var aoi = ee.FeatureCollection("users/mdgaines/HUC8_03130001_UpperChat");
print(aoi)

var jrc_monthly_col = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory');

// functions
function water_remapper(img){
  return(img.remap([2],[1],0));
}
  
function nonna_remapper(img){
  return(img.remap([1,2],[1,1],0));
}

function export_water(img, yr, szn, aoi){
  // ----------------------------------------------------------------------
  // Export image
  // // ----------------------------------------------------------------------
  Export.image.toDrive({
    image: img, 
    description: 'GSW_' + yr + '_' + szn,
    scale: 30,
    crs: 'EPSG:4326',  // WGS 84
    maxPixels: 1e13,
    region: aoi.geometry().bounds(),
  });
}

function get_szn_water(col){
  var water_col = col.map(water_remapper);
  var water_img = water_col.sum();
  
  var non_na_col = col.map(nonna_remapper);
  var non_na_img = non_na_col.sum();
  
  var fr = water_img.divide(non_na_img).multiply(100).int();
  var from_lst = ee.List.sequence(1,100);
  print(from_lst);
  var to_lst = ee.List.sequence(1,1,1,100);
  print(to_lst);
  
  var jrc_szn = fr.remap(from_lst,to_lst,0).clip(aoi).int();
  
  return(jrc_szn);
}

// ----------------------------------------------------------------------
// 2017
// ----------------------------------------------------------------------
var jrc_szn_col1 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(3,5,'month'))
  .filter(ee.Filter.calendarRange(2017,2017,'year'))
  .filterBounds(aoi);
  
var jrc_szn_col2 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(6,8,'month'))
  .filter(ee.Filter.calendarRange(2017,2017,'year'))
  .filterBounds(aoi);
  
var jrc_szn_col3 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(9,11,'month'))
  .filter(ee.Filter.calendarRange(2017,2017,'year'))
  .filterBounds(aoi);

var jrc_szn_col4 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(12,2,'month'))
  .filterDate('2017-12-01', '2018-03-01')
  .filterBounds(aoi);


var jrc_szn1 = get_szn_water(jrc_szn_col1);
// export_water(jrc_szn1, '2017', 'SPRING', aoi);

var jrc_szn2 = get_szn_water(jrc_szn_col2);
export_water(jrc_szn2, '2017', 'SUMMER', aoi);

var jrc_szn3 = get_szn_water(jrc_szn_col3);
export_water(jrc_szn3, '2017', 'FALL', aoi);

var jrc_szn4 = get_szn_water(jrc_szn_col4);
export_water(jrc_szn4, '2017', 'WINTER', aoi);

// print(jrc_szn);



// ----------------------------------------------------------------------
// 2019
// ----------------------------------------------------------------------
var jrc_szn_col1_19 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(3,5,'month'))
  .filter(ee.Filter.calendarRange(2019,2019,'year'))
  .filterBounds(aoi);
  
var jrc_szn_col2_19 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(6,8,'month'))
  .filter(ee.Filter.calendarRange(2019,2019,'year'))
  .filterBounds(aoi);
  
var jrc_szn_col3_19 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(9,11,'month'))
  .filter(ee.Filter.calendarRange(2019,2019,'year'))
  .filterBounds(aoi);

var jrc_szn_col4_19 = jrc_monthly_col.filter(
  ee.Filter.calendarRange(12,2,'month'))
  .filterDate('2019-12-01', '2020-03-01')
  .filterBounds(aoi);


var jrc_szn1_19 = get_szn_water(jrc_szn_col1_19);
export_water(jrc_szn1_19, '2019', 'SPRING', aoi);

var jrc_szn2_19 = get_szn_water(jrc_szn_col2_19);
export_water(jrc_szn2_19, '2019', 'SUMMER', aoi);

var jrc_szn3_19 = get_szn_water(jrc_szn_col3_19);
export_water(jrc_szn3_19, '2019', 'FALL', aoi);

var jrc_szn4_19 = get_szn_water(jrc_szn_col4_19);
export_water(jrc_szn4_19, '2019', 'WINTER', aoi);

// ----------------------------------------------------------------------
// Viz
// ----------------------------------------------------------------------
var viz = {
  min: 0.0,
  max: 1.0,
  palette: ['ffffff', 'fffcb8', '0905ff']
};

var viz2 = {
  min: 0.0,
  max: 2.0,
  palette: ['ffffff', 'fffcb8', '0905ff']
};

Map.centerObject(aoi);

// Map.addLayer(jrc_szn_col4.filter('month == 12'), viz2, 'dec')
// Map.addLayer(jrc_szn_col4.filter('month == 1'), viz2, 'jan')
// Map.addLayer(jrc_szn_col4.filter('month == 2'), viz2, 'feb')
// print(jrc_szn_col4)

// var dec = ee.Image('JRC/GSW1_4/MonthlyHistory/2017_12');
// Map.addLayer(dec, viz2, 'JRC Dec')
// var jan = ee.Image('JRC/GSW1_4/MonthlyHistory/2018_01');
// Map.addLayer(jan, viz2, 'JRC Jan')
// var feb = ee.Image('JRC/GSW1_4/MonthlyHistory/2018_02');
// Map.addLayer(feb, viz2, 'JRC Feb')

Map.addLayer(jrc_szn1, viz, 'Spring 2017');
Map.addLayer(jrc_szn2, viz, 'Summer 2017');
Map.addLayer(jrc_szn3, viz, 'Fall 2017');
Map.addLayer(jrc_szn4, viz, 'Winter 2017');

Map.addLayer(jrc_szn1_19, viz, 'Spring 2019');
Map.addLayer(jrc_szn2_19, viz, 'Summer 2019');
Map.addLayer(jrc_szn3_19, viz, 'Fall 2019');
Map.addLayer(jrc_szn4_19, viz, 'Winter 2019');

// Map.addLayer(jrc_szn_col1,'','Spring')
// Map.addLayer(jrc_szn_col2,'','Summer')
// Map.addLayer(jrc_szn_col3,'','Fall')
// Map.addLayer(jrc_szn_col4,'','Winter')



// ----------------------------------------------------------------------
// Viz GSW Monthly images
// ----------------------------------------------------------------------

// Define arguments for the getFilmstripThumbURL function parameters.
var filmArgs = {
  // dimensions: 128,
  region: bbox,
  format: 'png',
  // crs: 'EPSG:4326',
  min: 0,
  max: 2,
  palette: ['ffffff', 'fffcb8', '0905ff']
};

// ----------------------------------------------------------------------
// 2017
// ----------------------------------------------------------------------
// print('2017');
// // Print a URL that will produce the filmstrip when accessed.
// var clip_col1 = jrc_szn_col1.map(function(image){return image.clip(aoi)});
// print(clip_col1);
// print('spring');
// print(clip_col1.getFilmstripThumbURL(filmArgs));

// var clip_col2 = jrc_szn_col2.map(function(image){return image.clip(aoi)});
// print('summer');
// print(clip_col2.getFilmstripThumbURL(filmArgs));

// var clip_col3 = jrc_szn_col3.map(function(image){return image.clip(aoi)});
// print('fall');
// print(clip_col3.getFilmstripThumbURL(filmArgs));

// var clip_col4 = jrc_szn_col4.map(function(image){return image.clip(aoi)});
// print('winter');
// print(clip_col4.getFilmstripThumbURL(filmArgs));


// // ----------------------------------------------------------------------
// // 2019
// // ----------------------------------------------------------------------
// print('2019');
// // Print a URL that will produce the filmstrip when accessed.
// var clip_col1_19 = jrc_szn_col1_19.map(function(image){return image.clip(aoi)});
// print('spring');
// print(clip_col1.getFilmstripThumbURL(filmArgs));

// var clip_col2_19 = jrc_szn_col2_19.map(function(image){return image.clip(aoi)});
// print('summer');
// print(clip_col2_19.getFilmstripThumbURL(filmArgs));

// var clip_col3_19 = jrc_szn_col3_19.map(function(image){return image.clip(aoi)});
// print('fall');
// print(clip_col3_19.getFilmstripThumbURL(filmArgs));

// var clip_col4_19 = jrc_szn_col4_19.map(function(image){return image.clip(aoi)});
// print('winter');
// print(clip_col4_19.getFilmstripThumbURL(filmArgs));