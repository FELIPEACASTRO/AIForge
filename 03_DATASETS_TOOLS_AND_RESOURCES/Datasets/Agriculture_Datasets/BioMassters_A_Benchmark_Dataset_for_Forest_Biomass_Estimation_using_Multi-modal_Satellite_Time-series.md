# BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

## Description

BioMassters is a benchmark dataset created to investigate the potential of multimodal satellite time-series (Sentinel-1 SAR and Sentinel-2 MSI) for estimating Above-Ground Biomass (AGB) at large scale. The dataset uses open airborne LiDAR data from the Finnish Forest Centre as ground-truth reference. It was released as part of a machine learning challenge on the DrivenData platform in 2023. The goal is to advance the development of deep learning models to produce accurate, high-resolution biomass maps, overcoming the limitations of traditional field and LiDAR techniques, which are expensive and difficult to scale.

## Statistics

Location: Finland. Number of Samples: Nearly 13,000 forest patches (chips). Patch Size: 2,560 x 2,560 meters, resized to 256 x 256 pixels at 10-meter resolution. Collection Period: 2016 to 2021. Ground Reference (Label): Per-pixel Above-Ground Biomass (AGB), derived from airborne LiDAR measurements. Input Data: Monthly Sentinel-1 (SAR) and Sentinel-2 (MSI) time-series.

## Features

Multimodal time-series data: Combines Synthetic Aperture Radar (SAR) data from Sentinel-1 and multispectral imagery (MSI) from Sentinel-2. Spatial resolution of 10 meters. Temporal coverage of 5 years (2016 to 2021), with monthly images aggregated into a 12-month period (September to August) for each chip. The target (label) is the per-pixel AGB estimate (10x10m) within each chip.

## Use Cases

Biomass Modeling: Training and evaluation of machine learning and deep learning models for AGB estimation in forests. Carbon Monitoring: Use in forest inventories and monitoring of carbon sequestration capacity. Remote Sensing Research: Investigation of the predictive value of multimodal data (SAR and MSI) and dense time-series for environmental applications. ML Competitions: Serves as a benchmark dataset for data science and machine learning challenges.

## Integration

The dataset and associated code are available on the project website and GitHub repository. Direct access to the data is provided through the Source Cooperative (mentioned in the TorchGeo documentation, which offers an interface to the dataset). The code of the winning competitors of the DrivenData challenge is available on GitHub, serving as examples of implementation and feature engineering.

**Access Example (via TorchGeo, indicating the original source):**
The dataset can be accessed programmatically through libraries such as TorchGeo, which points to the Source Cooperative as the download source.

**File Structure:**
The data is provided as GeoTIFFs, with the filename following the format `{chip_id}_{satellite}_{month_number}.tif`.
- `S1`: Sentinel-1 (SAR)
- `S2`: Sentinel-2 (MSI)
- `month_number`: 00 (September) to 11 (August)

**Feature Engineering:**
Feature engineering approaches focus on extracting temporal and multimodal information, such as vegetation indices (NDVI, EVI) from Sentinel-2 and polarization and backscatter metrics (VV, VH) from Sentinel-1, in addition to temporal statistics (means, deviations, peaks) throughout the year.

## URL

https://nascetti-a.github.io/BioMasster/
