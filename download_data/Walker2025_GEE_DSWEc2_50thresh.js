// ===============================================
// DSWE 2.0 Landsat Collection 2 - Monthly Composites
// ===============================================
//
// DSWE_LANDSATc2_monthly_composites.js
//
// This script creates monthly composites of Dynamic Surface Water Extent (DSWE) v2.0 products: 
// categories of ground surface inundation as detected in cloud-/shadow-/snow-free 
// Landsat pixels. Where multiple values are returned for pixels within a month,
// higher-confidence water categories take precedence over lower/partial ones, while water 
// or no-water categories dominate cloudy pixels. Algorithm and approach were originally 
// derived from JJones's 2015 DSWE Algorithm Description pdf and updated to match changes in 
// June 2018 DSWE v2.0 documentation.
//
// DSWE v1.0 coding: Jessica Walker 
// DSWE v2.0 coding: Jason Kreitler, Roy Petrakis, Chris Soulard, Jessica Walker
// Update to Landsat Collection 2: Jessica Walker
//
// USGS contacts: Chris Soulard, csoulard@usgs.gov; Jessica Walker, jjwalker@usgs.gov
//
//
// Citations: 
//
//  Jones, 2015. Efficient Wetland Surface Water Detection and Monitoring via Landsat: 
//              Comparison with in situ Data from the Everglades Depth Estimation Network.
//              Remote Sensing 7(9). 12503-12538; https://doi.org/10.3390/rs70912503
//
//  Landsat Dynamic Surface Water Extent (DSWE) Product Guide. Collection 1.
//              https://www.usgs.gov/media/files/landsat-dynamic-surface-water-extent-product-guide
//
//  Landsat Dynamic Surface Water Extent ADD. Collection 1.
//              https://www.usgs.gov/media/files/landsat-dynamic-surface-water-extent-add
//
//  Walker JJ, Waller EK, Kreitler JR, Petrakis RE, Soulard CE. (2025). DSWE_GEE 
//              v2.0.0 [Software release]. U.S. Geological Survey. DOI: 
//              https://doi.org/10.5066/P14CF2B2
//
// -------------------------------------------
// Input: User-supplied date range (within Landsat TM availability)
// Output: Single multi-band image of monthly DSWE composites as a GEE Asset
// Output file name is automatically set to export with the name
// "DSWE_<startyear>_<startmonth>_to_<endyear>_<endmonth>_monthly_images"
// but can be changed when starting the task in GEE.
//        
// DSWE categories:
// 0 - Not Water
// 1 - Water - High Confidence
// 2 - Water - Moderate Confidence
// 3 - Partial Surface Water Pixel
// 4 - Water or wetland, low confidence
// 9 - Cloud, Cloud Shadow, or Snow  
// null - Fill (no data) ** currently left masked
//
// ----------------------------------------------------------------------
// Notes:
//
// This script uses the Albers Equal Conic projection (EPSG 5070) which
// is a proxy for the Albers projection used for the Landsat ARD product. 
//
// Landsat data are from Collection 2. 
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------------------------
// This software is in the public domain because it contains materials that
// originally came from the United States Geological Survey, an agency of the
// United States Department of Interior. For more information, see the official
// USGS copyright policy at http://www.usgs.gov/visual-id/credit_usgs.html#copyright
// --------------------------------------------------------------------------------------- 

// ----------------- Input user-required info ---------------------------
// Dates should be within Landsat TM range (Aug 22, 1982 to present)
// Note that all dates will be automatically standardized to the first of the 
// given startdate month and the end of the enddate month in order to return 
// a series of full-month composites
var startdate = ee.Date('2019-03-01');
var enddate = ee.Date('2019-06-01');
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Define AOI 
// ----------------------------------------------------------------------
// User can specify their own aoi by using the drawing tool and uncommenting var aoi = geometry
var aoi = ee.FeatureCollection("users/mdgaines/HUC8_03130001_UpperChat");

//var aoi = geometry;
//
// ----------------------------------------------------------------------
// Import DEM (USGS National Elevation Dataset 1/3 arc-second)
// ----------------------------------------------------------------------

var dem = ee.Image("USGS/NED");

// ---------------------------------------------------------------------
// Apply scaling factor to optical bands
// ---------------------------------------------------------------------
function scaleBands(img) {
  return img.addBands(img.select('SR_B.').multiply(0.0000275).add(-0.2), null, true);
}

// ----------------------------------------------------------------------
// Mask clouds, cloud shadows, and snow
// ----------------------------------------------------------------------

function maskClouds(img) {
  // Bit 0 - Fill
  // Bit 1 - Dilated Cloud
  // Bit 2 - Unused
  // Bit 3 - Cloud
  // Bit 4 - Cloud Shadow
  // Bit 5 - Snow
  var qaMask = img.select('QA_PIXEL').bitwiseAnd(parseInt('111101', 2)).neq(0);
  var saturationMask = img.select('QA_RADSAT').eq(0);
  return img.addBands(qaMask.rename('clouds'));
    //  .updateMask(qaMask)
    //  .updateMask(saturationMask);
}

// ----------------------------------------------------------------------
// Load Landsat imagery 
// ----------------------------------------------------------------------

// Define Landsat surface reflectance bands
var sensor_band_dict = ee.Dictionary({
                        l9:  ee.List([1,2,3,4,5,6,17,18]),
                        l8 : ee.List([1,2,3,4,5,6,17,18]),
                        l7 : ee.List([0,1,2,3,4,5,17,18]),
                        l5 : ee.List([0,1,2,3,4,5,17,18]),
                        l4 : ee.List([0,1,2,3,4,5,17,18])  
                        });
// Sensor band names corresponding to selected band numbers                        
var bandNames = ee.List(['blue','green','red','nir','swir1','swir2','pixel_qa','radsat_qa']);

// ------------------------------------------------------
// Landsat 4 - Data availability Aug 22, 1982 - Dec 14, 1993
var ls4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
            .filterBounds(aoi)
            .map(scaleBands)
            .map(maskClouds)
            .select(ee.List(sensor_band_dict.get('l4')).cat(['clouds']), bandNames.cat(['clouds']));

// ------------------------------------------------------
// Landsat 5 - Data availability Jan 1, 1984 - May 5, 2012
var ls5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
            .filterBounds(aoi)
            .map(scaleBands)
            .map(maskClouds)
            .select(ee.List(sensor_band_dict.get('l5')).cat(['clouds']), bandNames.cat(['clouds']));

// Landsat 7 data are only used during operational SLC and
// to fill the gap between the end of LS5 and the beginning
// of LS8 data collection

// Prior to SLC-off            
// -------------------------------------------------------
// Landsat 7 - Data availability Jan 1, 1999 - Aug 9, 2016
// SLC-off after 31 May 2003
var ls7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') 
              .filterDate('1999-01-01', '2003-05-31') 
              .filterBounds(aoi)
              .map(scaleBands)
              .map(maskClouds)
              .select(ee.List(sensor_band_dict.get('l7')).cat(['clouds']), bandNames.cat(['clouds']));
        
// Post SLC-off; fill the LS 5 gap
// -------------------------------------------------------
// Landsat 7 - Data availability Jan 1, 1999 - Aug 9, 2016
// SLC-off after 31 May 2003
var ls7_2 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') 
              .filterDate('2012-05-05', '2014-04-11') 
              .filterBounds(aoi)
              .map(scaleBands)
              .map(maskClouds)
              .select(ee.List(sensor_band_dict.get('l7')).cat(['clouds']), bandNames.cat(['clouds']));
         
// --------------------------------------------------------
// Landsat 8 - Data availability Apr 11, 2014 - present
var ls8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')  
            .filterBounds(aoi)
            .map(scaleBands)
            .map(maskClouds)
            .select(ee.List(sensor_band_dict.get('l8')).cat(['clouds']), bandNames.cat(['clouds']));

// --------------------------------------------------------
// Landsat 9 - Data availability 31 Oct 2021 - present
var ls9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')  
              .filterBounds(aoi)//.geometry().bounds())
              .map(scaleBands)
              .map(maskClouds)
              .select(ee.List(sensor_band_dict.get('l9')).cat(['clouds']), bandNames.cat(['clouds']));
              
            
// Merge landsat collections
var ic_merged = ee.ImageCollection(ls4
                .merge(ls5)
                .merge(ls7)
                .merge(ls7_2)
                .merge(ls8)
                .merge(ls9).sort('system:time_start'))
                .filterDate(startdate, enddate);

// print(ic_merged);

// ----------------------------------------------------------------------
// Calculate hillshade mask
// ----------------------------------------------------------------------
function addHillshade(img) {
    var solar_azimuth = img.get('SUN_AZIMUTH');
    var solar_elevation = img.get('SUN_ELEVATION'); 
   return img.addBands(ee.Terrain.hillshade(dem, solar_azimuth, solar_elevation).rename('hillshade')); 
}

// Add hillshade bands
var ic_hillshade = ic_merged.map(addHillshade);


// ----------------------------------------------------------------------
// Calculate DSWE indices
// ----------------------------------------------------------------------
function addIndices(img){
// NDVI (Normalized Difference Vegetation Index) = (NIR - Red)/(NIR + Red)
    img = img.addBands(img.normalizedDifference(['nir', 'red']).select([0], ['ndvi']));

// MNDWI (Modified Normalized Difference Wetness Index) = (Green - SWIR1) / (Green + SWIR1)
    img = img.addBands(img.normalizedDifference(['green', 'swir1']).select([0], ['mndwi']));

// MBSRV (Multi-band Spectral Relationship Visible) = Green + Red
    img = img.addBands(img.select('green').add(img.select('red')).select([0], ['mbsrv'])).toFloat();

// MBSRN (Multi-band Spectral Relationship Near-Infrared) = NIR + SWIR1
    img = img.addBands(img.select('nir').add(img.select('swir1')).select([0], ['mbsrn']).toFloat());

// AWEsh (Automated Water Extent Shadow) = Blue + (2.5 * Green) + (-1.5 * mbsrn) + (-0.25 * SWIR2)
    img = img.addBands(img.expression('blue + (2.5 * green) + (-1.5 * mbsrn) + (-0.25 * swir2)', {
         'blue': img.select('blue'),
         'green': img.select('green'),
         'mbsrn': img.select('mbsrn'),
         'swir2': img.select('swir2')
    }).select([0], ['awesh'])).toFloat();     
    return img;
}

// Add indices
var ic_indices = ic_hillshade.map(addIndices);


// ----------------------------------------------------------------------
// DSWE parameter testing
// ----------------------------------------------------------------------

// Bitmask of 11111 = 16 + 8 + 4 + 2 + 1 = 31 = 1F in hex 

// 1. ========== Function: test MNDWI ===========
// If (MNDWI > 0.124) set the ones digit (i.e., 00001)
function test_mndwi(img) {
  var mask = img.select('mndwi').gt(0.124);
  return img.addBands(mask
            .bitwiseAnd(0x1F)  
            .rename('mndwi_bit'));
}

// 2. ======== Function: compare MBSRV and MBSRN ========
// If (MBSRV > MBSRN) set the tens digit (i.e., 00010)
function test_mbsrv_mbsrn(img) {
  var mask = img.select('mbsrv').gt(img.select('mbsrn'));
  return img.addBands(mask
            .bitwiseAnd(0x1F) 
            .leftShift(1)  // shift left 1 space
            .rename('mbsrn_bit'));   
}

// 3. ======== Function: test AWEsh ========
// If (AWEsh > 0.0) set the hundreds digit (i.e., 00100)
function test_awesh(img) {
  var mask = img.select('awesh').gt(0.0);
  return img.addBands(mask
              .bitwiseAnd(0x1F)
              .leftShift(2)  // shift left 2 spaces
              .rename('awesh_bit'));  
}

// 4. ======= Function: test PSW1 ========
// If (MNDWI > -0.44 && SWIR1 < 0.09 && NIR < 0.15 & NDVI < 0.7) set the thousands digit (i.e., 01000)
function test_mndwi_swir1_nir(img) {
  var mask = img.select('mndwi').gt(-0.44)
           .and(img.select('swir1').lt(0.09))
              .and(img.select('nir').lt(0.15))
              .and(img.select('ndvi').lt(0.7));
  return img.addBands(mask            
            .bitwiseAnd(0x1F)
            .leftShift(3)  // shift left 3 spaces
            .rename('swir1_bit')); 
}

// 5. ======= Function: test PSW2 =========
// If (MNDWI > -0.5 && SWIR1 < 0.3 && SWIR2 < 0.1 && NIR < 0.25 && Blue < 0.1) set the ten-thousands digit (i.e., 10000)
function test_mndwi_swir2_nir(img){
  var mask = img.select('mndwi').gt(-0.5)
              .and(img.select('swir1').lt(0.30))
              .and(img.select('swir2').lt(0.1))
              .and(img.select('nir').lt(0.25))
              .and(img.select('blue').lt(0.1));
  return img.addBands(mask
              .bitwiseAnd(0x1F)
              .leftShift(4)  // shift left 4 spaces
              .rename('swir2_bit'));  
}

// Function: consolidate individual bit bands
function sum_bit_bands(img){
  var bands = img.select(['mndwi_bit', 'mbsrn_bit', 'awesh_bit', 'swir1_bit', 'swir2_bit']);
  var summed_bands = bands.reduce(ee.Reducer.bitwiseOr());
  return img.addBands(summed_bands.rename('summed_bit_band'));
}

// Add individual bit bands to image collection and summarize
var ic_indices_bit = ee.ImageCollection(ic_indices)
              .map(test_mndwi)
              .map(test_mbsrv_mbsrn)
              .map(test_awesh)
              .map(test_mndwi_swir1_nir)
              .map(test_mndwi_swir2_nir)
              .map(sum_bit_bands);

// ----------------------------------------------------------------------
// Produce DSWE layers
// ----------------------------------------------------------------------

// Construct slope image from DEM
var slope = ee.Terrain.slope(dem);

// Remap based on JJones key
var ic_indices_remap = ic_indices_bit.map(function(img){
  var reclass = img.select('summed_bit_band').remap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                            30, 31],
                                            
                                            [0, 0, 0, 4, 0, 4, 4, 2, 0, 4,
                                             4, 2, 4, 2, 2, 1, 4, 4, 4, 2,
                                             4, 2, 2, 1, 3, 2, 2, 1, 2, 1, 
                                             1, 1]).rename('dswe');
                                             
// ID cloud-contaminated pixels                                           
  reclass = reclass.where(img.select('clouds').eq(1), 9);  
// ID shaded areas
  reclass = reclass.where(img.select('hillshade').lte(110), 8);
// ID slopes
  reclass = reclass.where(img.select('dswe').eq(4) && slope.gte(5.71).or  // 10% slope = 5.71°
              (img.select('dswe').eq(3) && slope.gte(11.31)).or           // 20% slope = 11.31°
              (slope.gte(16.7)), 0);                                      // 30% slope = 16.7°

  return img.addBands(reclass).select('dswe');
});

// print('img indices remap ', ic_indices_remap.limit(10));
print('img indices',ic_indices_remap.size());

// Test calculating percent of high-conf water
var dswe_c1 = ic_indices_remap.map(function(img) {
  return img.updateMask(img.select('dswe').eq(1));
});
var dswe_c1_sum = dswe_c1.count();
// print(dswe_c1_sum);
Map.addLayer(dswe_c1_sum, {min: 1, max: 21}, 'DSWE C1 Count');

var dswe_c2 = ic_indices_remap.map(function(img) {
  return img.updateMask(img.select('dswe').lte(2).and(img.select('dswe').gt(0)));
});
var dswe_c2_sum = dswe_c2.count();
// print(dswe_c2_sum);
Map.addLayer(dswe_c2_sum, {min: 1, max: 21}, 'DSWE C2 Count');

var dswe_c3 = ic_indices_remap.map(function(img) {
  return img.updateMask(img.select('dswe').lte(3).and(img.select('dswe').gt(0)));
});
var dswe_c3_sum = dswe_c3.count();
// print(dswe_c3_sum);
Map.addLayer(dswe_c3_sum, {min: 1, max: 21}, 'DSWE C3 Count');

// Calculate sum of valid pixels (pixels that are not clouds (!=9))
var non_cloud = ic_indices_remap.map(function(img) {
  return img.updateMask(img.select('dswe').lt(9));
});
var non_cloud_sum = non_cloud.count();
// print(dswe_c3_sum);
Map.addLayer(non_cloud_sum, {min: 1, max: 21}, 'Non-cloud Count');

//// cloud count (just to double check our non-cloud count)
// var cloud = ic_indices_remap.map(function(img) {
//   return img.updateMask(img.select('dswe').eq(9));
// });
// var cloud_sum = cloud.count();
// // print(dswe_c3_sum);
// Map.addLayer(cloud_sum, {min: 1, max: 21}, 'Cloud Count');

// Thresholding at 50% C1 occurance for all valid occurances
var p_dswe_c1 = dswe_c1_sum.divide(non_cloud_sum).multiply(100);
// Map.addLayer(p_dswe_c1, {min: 0, max: 100}, 'Percent C1');
var p50_dswe_c1 = p_dswe_c1.gt(50);
Map.addLayer(p50_dswe_c1, {min: 0, max: 1}, 'Percent C1 > 50%');

// Thresholding at 50% C2 occurance for all valid occurances
var p_dswe_c2 = dswe_c2_sum.divide(non_cloud_sum).multiply(100);
// Map.addLayer(p_dswe_c2, {min: 0, max: 100}, 'Percent C2');
var p50_dswe_c2 = p_dswe_c2.gt(50);
Map.addLayer(p50_dswe_c2, {min: 0, max: 1}, 'Percent C2 > 50%');

// Thresholding at 50% C3 occurance for all valid occurances
var p_dswe_c3 = dswe_c3_sum.divide(non_cloud_sum).multiply(100);
// Map.addLayer(p_dswe_c3, {min: 0, max: 100}, 'Percent C3');
var p50_dswe_c3 = p_dswe_c3.gt(50);
Map.addLayer(p50_dswe_c3, {min: 0, max: 1}, 'Percent C3 > 50%');


Export.image.toDrive({
  image: p50_dswe_c1, 
  description: 'DSWEc2_Spring2019_50thresh_conf1',
  crs: 'EPSG:5070',  // USGS Albers Equal Area Conic
  crsTransform: [30,0,-2265585,0,-30,2114805], // transform from EROS projection
  maxPixels: 1e13,
  region: aoi.bounds(),  // bounds is necessary for large or complex regions
});

Export.image.toDrive({
  image: p50_dswe_c2, 
  description: 'DSWEc2_Spring2019_50thresh_conf2',
  crs: 'EPSG:5070',  // USGS Albers Equal Area Conic
  crsTransform: [30,0,-2265585,0,-30,2114805], // transform from EROS projection
  maxPixels: 1e13,
  region: aoi.bounds(),  // bounds is necessary for large or complex regions
});

Export.image.toDrive({
  image: p50_dswe_c3, 
  description: 'DSWEc2_Spring2019_50thresh_conf3',
  crs: 'EPSG:5070',  // USGS Albers Equal Area Conic
  crsTransform: [30,0,-2265585,0,-30,2114805], // transform from EROS projection
  maxPixels: 1e13,
  region: aoi.bounds(),  // bounds is necessary for large or complex regions
});
