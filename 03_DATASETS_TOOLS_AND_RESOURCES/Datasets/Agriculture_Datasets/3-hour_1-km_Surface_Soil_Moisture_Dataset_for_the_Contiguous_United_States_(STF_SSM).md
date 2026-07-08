# 3-hour, 1-km Surface Soil Moisture Dataset for the Contiguous United States (STF_SSM)

## Description

A spatially seamless surface soil moisture (SSM) dataset at 3-hour, 1-km resolution for the Contiguous United States (CONUS), developed using a spatio-temporal fusion method based on virtual image pairs. The dataset, named STF_SSM, combines the advantages of the SMAP L4 SSM product (high temporal resolution of 3 hours, but 9-km spatial resolution) and the Crop-CASMA dataset (high spatial resolution of 1 km, but daily temporal resolution and with spatial gaps). The result is a long time-series dataset (2015-2023) with high spatial and temporal resolution, crucial for drought monitoring and the validation of land-surface models. The dataset was published in 2025, making it a very recent research resource.

## Statistics

- **Coverage Period:** 2015-2023.
- **Resolution:** 1 km (spatial) and 3 hours (temporal).
- **Validation:** Average Correlation Coefficient (CC) of 0.716 at the daily scale and 0.689 at the 3-hour scale, compared to in-situ data.
- **Error:** Unbiased Root Mean Square Error (ubRMSE) of 0.057 m³/m³ (daily) and 0.062 m³/m³ (3-hour).
- **Volume:** The dataset is composed of multiple data files (Child Items) per year, indicating a large volume of geospatial data.

## Features

- **Spatial Resolution:** 1 km (high resolution).
- **Temporal Resolution:** 3 hours (high temporal, intra-daily resolution).
- **Coverage:** Contiguous United States (CONUS).
- **Period:** April 1, 2015 to December 31, 2023.
- **Method:** Spatio-Temporal Fusion (STF) of SMAP L4 and Crop-CASMA data.
- **Characteristics:** Spatially seamless, filling data gaps.

## Use Cases

- **Drought Monitoring:** Provides insights into rapid changes in soil moisture, essential for detecting and monitoring droughts and wet periods.
- **Land-Surface Modeling:** A critical data source for the calibration and validation of hydrological and land-surface models.
- **Precision Agriculture:** The high spatial resolution (1 km) is valuable for precision agriculture applications, such as irrigation optimization and yield estimation.
- **Ecosystem Studies:** Enables the analysis of ecosystem responses to rapid variations in soil moisture.
- **Weather Forecasting:** Improvement of weather forecasting and extreme-event (flood) prediction.

## Integration

The STF_SSM dataset is available for download through the USGS ScienceBase catalog. The data is organized into "Child Items" by year. Primary access is via DOI and the ScienceBase page.

**DOI:** `https://doi.org/10.5066/P13CCN69`

**Access via ScienceBase:**
1. Navigate to the ScienceBase page.
2. Click the "Child Items" links to download the individual data files (usually in NetCDF or similar format, common for geospatial data).
3. The dataset generation code (STF method) is also available for reference and reproduction at: `https://doi.org/10.6084/m9.figshare.28188011` (Yang et al., 2025b).

*Note: Direct access to the data files requires navigating the ScienceBase page for the specific download links of each year.*

## URL

https://www.sciencebase.gov/catalog/item/67ace224d34e329fb2046073
