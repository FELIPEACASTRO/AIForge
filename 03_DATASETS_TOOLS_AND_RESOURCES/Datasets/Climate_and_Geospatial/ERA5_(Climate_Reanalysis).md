# ERA5 (Climate Reanalysis)

## Description
ERA5 is the fifth generation of reanalysis from the European Centre for Medium-Range Weather Forecasts (ECMWF) for global climate and weather. The dataset combines model data with observations from around the world (data assimilation) to produce a globally complete and consistent dataset spanning the last eight decades, with data available from 1940 onward. It provides hourly estimates for a large number of atmospheric, ocean-wave, and land-surface variables. It includes an uncertainty estimate sampled by an underlying 10-member ensemble.

## Statistics
**Temporal Coverage:** 1940 to the present (hourly data). **Horizontal Resolution (Reanalysis):** 0.25° x 0.25° (atmosphere) and 0.5° x 0.5° (ocean waves). **Temporal Resolution:** Hourly. **Update Frequency:** Daily (with a latency of about 5 days for the preliminary ERA5T version). **File Format:** GRIB (native), with optimized Zarr versions available for ML/AI. **Versions:** ERA5 (main), ERA5-Land (land component), ERA5T (preliminary), Anemoi training-ready version (optimized for ML, 1979-2023).

## Features
ERA5 offers a vast range of variables, including wind components (at 10m and 100m), temperature, precipitation, radiation, and many other land-surface, atmospheric, and ocean-wave quantities. The main dataset has a horizontal resolution of 0.25° x 0.25° for the atmosphere and 0.5° x 0.5° for ocean waves. It has a version optimized for Machine Learning (Anemoi) in Zarr format.

## Use Cases
ERA5 is widely used in climate research and monitoring, serving as the basis for weather and climate forecasting models. It is the preferred dataset for training Machine Learning (ML) and Artificial Intelligence (AI) models in weather forecasting and spatial downscaling. Other use cases include retrospective climate research, bias correction in evapotranspiration estimates, and socioeconomic impact studies.

## Integration
Primary access is provided through the Copernicus Climate Data Store (CDS), using the web interface or programmatically through the **CDS API service** (Python `cdsapi` library). Alternatively, the dataset is available on platforms such as Google Earth Engine (monthly aggregates, ERA5-Land) and AWS Open Data (in GRIB and cloud-optimized Zarr formats). Use of the CDS API is the recommended method for batch downloads.

## URL
[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
