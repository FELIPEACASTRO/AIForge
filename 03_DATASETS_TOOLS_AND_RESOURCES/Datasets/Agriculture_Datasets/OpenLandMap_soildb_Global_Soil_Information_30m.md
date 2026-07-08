# OpenLandMap-soildb: Global Soil Information at 30m

## Description

OpenLandMap-soildb is a global, dynamic soil information dataset with a spatial resolution of 30 meters, covering the period from 2000 to 2022 and including future projections. It uses Spatiotemporal Machine Learning (Quantile Regression Random Forest) trained on a vast compilation of legacy soil samples to provide consistent predictions of key soil variables. The main focus is mapping soil properties at depth (3D), with updates every 5 years for dynamic variables such as pH and Soil Organic Carbon (SOC). This resource is fundamental for precision agriculture and biomass studies.

## Statistics

**Spatial Resolution:** 30 meters. **Period:** 2000–2022+. **Training Samples:** 272,000 samples for soil pH in H2O; 363,000 samples for texture (clay, silt, sand); 216,000 samples for Soil Organic Carbon (SOC) density. **Validation Metrics (pH):** RMSE of 0.51 and CCC of 0.91. **Important Variables (pH):** CHELSA Aridity Index, annual precipitation, and degree of salinity. **Carbon Stock Estimate:** 461 Pg (Petagrams) for 0–30 cm (2020–2022+).

## Features

**Key Variables:** Soil pH in H2O, bulk density, soil texture fractions (clay, silt, and sand), Soil Organic Carbon (SOC), and soil types (USDA soil taxonomy subgroups). **Methodology:** Uses Spatiotemporal Machine Learning (Quantile Regression Random Forest) for global predictions. **Resolution:** 30-meter spatial resolution, compatible with medium- and fine-resolution satellite data (Landsat, Sentinel). **Coverage:** Global, with soil property mapping at 5-year intervals for dynamic variables.

## Use Cases

**Precision Agriculture:** Mapping management zones and optimizing the application of fertilizers and amendments (such as lime for pH correction). **Biomass and Carbon Modeling:** Predicting plant biomass and estimating Soil Organic Carbon (SOC) stocks for climate mitigation and carbon sequestration projects. **Environmental Studies:** Monitoring soil degradation and supporting land restoration projects at a global scale. **Hydrological Modeling:** Using texture fractions and bulk density to estimate water retention capacity and soil moisture.

## Integration

The resulting data products can be accessed and downloaded through the Zenodo repository, with DOI `10.5281/zenodo.15470431`. The training dataset is also available at `10.5281/zenodo.4748499`. Both are released under a CC-BY license. Integration into AI and Biomass projects is done by downloading the 30m *rasters* and incorporating them into *machine learning* models or GIS platforms. The source code and methodology are available on the OpenLandMap GitHub.

## URL

https://essd.copernicus.org/preprints/essd-2025-336/
