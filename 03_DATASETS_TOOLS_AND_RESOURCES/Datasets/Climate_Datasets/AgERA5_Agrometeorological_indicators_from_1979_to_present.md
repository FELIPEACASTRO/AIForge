# AgERA5: Agrometeorological indicators from 1979 to present

## Description

AgERA5 is a daily surface meteorological dataset, derived from the ECMWF ERA5 reanalysis, specifically tailored for applications in agriculture and agroecological studies. It provides essential agrometeorological variables, such as temperature, precipitation and humidity, at a spatial resolution of 0.1° x 0.1°, with global coverage. The dataset is corrected for finer topography, making it more accurate for agricultural modeling.

## Statistics

**Temporal Coverage:** From 1979 to present (daily updates).
**Temporal Resolution:** Daily.
**Spatial Resolution:** 0.1° x 0.1° (approximately 10 km).
**Geographic Coverage:** Global.
**Size:** Variable, depending on the time and area selection. Contains 19 agrometeorological variables.

## Features

Daily surface meteorological data (10m wind speed, 2m dew point temperature and humidity, 2m relative humidity, 2m temperature, precipitation, solar radiation, etc.). Spatial resolution of 0.1° x 0.1° (approximately 10 km). Correction for fine topography. Available in NetCDF-4 format.

## Use Cases

**Crop Modeling:** Provides essential input data for crop growth and yield models (e.g., DSSAT, APSIM).
**Drought Monitoring:** Used to compute drought indices and monitor water stress in agriculture.
**Climate Impact Assessment:** Analysis of how climate change affects agricultural productivity.
**Agrometeorological Forecasting:** Basis for agricultural advisory services and short- and long-term forecasts.

## Integration

AgERA5 is available through the Copernicus Climate Data Store (CDS). Access is provided via an interactive web download form or programmatically using the CDS API (Python). Libraries such as `ag5Tools` (R) also facilitate download and extraction.

Example of access via the CDS API (Python):
```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-agrometeorological-indicators',
    {
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'total_precipitation',
        ],
        'year': '2024',
        'month': '01',
        'day': [
            '01', '02', '03',
        ],
        'time_aggregation': 'daily_mean',
        'area': [
            -10, -50, -30, -30, # [N, W, S, E]
        ],
    },
    'download.nc')
```

## URL

https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview