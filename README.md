# PlanetBasemaps_DSWE_comparison

This repository contains code for the classification of PlanetBasemap imagery to produce seasonal, 4.77 m resolution, surface water extent maps for 2017 and 2019 in the Upper Chattahoochie River Watershed (HUC08 03130001). The seasonal surface water classifications are available on Zenodo: [linked here](10.5281/zenodo.13338427).

This repository also contains the code for the analysis supporting the manuscript "Impact of spatial scale on optical Earth observation-derived seasonal surface water extents" accepted for publication in *Geophysical Research Letters* (doi: ).

---

## Folder Breakdown

### > download_data
* Download PlanetBasemap data (using a Planet Labs internal module).
* Download seasonal USGS Dynamic Surface Water Extent (DSWE) data from Google Earth Engine (GEE) ([Walker et al., 2025](https://doi.org/10.5066/P13UQSZN)).
* Download seasonal Global Surface Water (GSW) data from GEE ([Pekel et al., 2016](https://www.nature.com/articles/nature20584)).

### > random_forest
* Train, test, and save a random forest model.
* Apply a random forest model to PlanetBasemap imagery.
* Conduct and accuracy assessment of the random forest model.

### > results_analysis
* Compile classified PlanetBasemap imagery into seasonal composites.
* Compare PlanetBasemap seasonal surface water classifications with DSWE and GSW seasonal classifications.
* Conduct a mixed pixel analysis by downscaling PlanetBasemap to 30 m resolution.
* Assess the surface water area classified by PlanetBasemaps, DSWE, and GSW at the HUC12 scale and within National Wetland Inventory delineated Lakes, Rivers, and Ponds.

### > visualization
* Plot various results.
* Animate differences in PlanetBasemap and DSWE seasonal surface water classifications.

---

## Example comparisons of PlanetBasemap and DSWE

![PlanetBasemap vs DSWE in a forested headwater area.](./visualization/box1_20250916-170054.gif)

---

## Set-up

Activate dswe-ps env

Set your Planet API key as a variable in your environment

`conda env config vars set PL_API_KEY={YOUR PLANET API KEY}`

Download PlanetBasemap, DSWE, and GSW data.

* PlanetBasemap data is commerically available from Planet Labs.
* DSWE data is publically available from GEE using code from [Walker et al., 2025](https://doi.org/10.5066/P13UQSZN)
* GSW data is available from GEE (https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_MonthlyHistory)