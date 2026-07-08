# Topographic Factors (Elevation, Slope, Aspect) as Features

## Description

Topographic characteristics derived from Digital Elevation Models (DEMs), such as Elevation, Slope, and Aspect, are used as static environmental features to improve the accuracy of Machine Learning models (e.g., Random Forest) in estimating Aboveground Biomass (AGB) and predicting crop yield. These factors help explain the spatial heterogeneity and microclimatic conditions that affect vegetation growth and the distribution of nutrients/water.

## Statistics

The inclusion of topographic factors (slope and aspect) in rubber tree biomass estimation models resulted in an average increase of 0.007 in R², an average absolute reduction of 4.54 t/ha in Bias, and an average reduction of 9.73% in RMSE. Elevation and the 30m SRTM DEM are frequently cited as data sources.

## Features

Elevation (Altitude), Slope (Degree of incline), Aspect (Slope orientation). These are static features that influence the microclimate, soil erosion, water and nutrient distribution, and solar radiation incidence.

## Use Cases

Aboveground Biomass (AGB) estimation in forests and plantations (e.g., rubber tree, mountain forests). Crop yield prediction (e.g., sugarcane, maize, wheat). Mapping of soil properties and monitoring of agricultural land abandonment. Modeling of soil loss by water erosion (RUSLE).

## Integration

Topographic data are typically derived from DEMs (such as SRTM, ASTER GDEM, or high-resolution DEMs from UAV/LiDAR) using GIS tools (e.g., ArcGIS, QGIS, Google Earth Engine - GEE). GEE is cited as a source for the 30m SRTM DEM. The features are then integrated as input variables in Machine Learning models (e.g., Random Forest, XGBoost, Deep Learning) together with remote sensing data (e.g., vegetation indices, multispectral data).

## URL

https://www.sciencedirect.com/science/article/pii/S2666719325001955