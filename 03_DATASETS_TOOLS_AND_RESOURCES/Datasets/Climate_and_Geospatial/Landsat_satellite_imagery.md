# Landsat (Satellite Imagery)

## Description
The Landsat program is a series of Earth observation satellite missions, jointly managed by NASA and the U.S. Geological Survey (USGS). Since 1972, Landsat satellites have continuously acquired images of the Earth's surface, providing an uninterrupted data archive. This archive is considered the "gold standard" for medium-resolution (30 meters) satellite data due to its rigorous and documented calibration. The data have been open and free since 2008, which has driven the development of new products and applications, especially in Machine Learning and Artificial Intelligence. The program is fundamental for monitoring changes in the Earth's surface, natural resources, and the environment.

## Statistics
**Daily Volume:** About **1 terabyte (TB)** of new Landsat mission data is acquired and downlinked daily. After processing, about **3 TB** are added to the USGS EROS archive per day. **Versions:** The data are organized into **Collections**, with **Collection 2** being the current operational version (introduced in December 2020), which includes Level 1 and Level 2 products (Surface Reflectance and Surface Temperature). **Samples/Scenes:** The total archive is vast, with the number of Collection 2 Level 1 and Level 2 scenes growing continuously through the Global Archive Consolidation (LGAC) program. The total volume of data downloaded from the USGS EROS archive from December 2008 (FY 2009) to July 2025 (FY 2025) is measured in petabytes.

## Features
**Spatial Resolution:** Medium (30 meters for most bands). **Coverage:** Global and continuous since 1972 (long time series). **Spectral Bands:** Ranges from 7 to 11 bands (depending on the satellite, such as Landsat 8 and 9), including visible, near-infrared (NIR), and thermal infrared (TIRS). **Products:** Includes Level 1 (scene-based) products, Level 2 (Surface Reflectance and Surface Temperature), and U.S. Analysis Ready Data (ARD). **Access:** Free and open. **Calibration:** Considered the "gold standard" for medium-resolution data due to its rigorous calibration.

## Use Cases
**Environmental Monitoring:** Mapping and monitoring of changes in land cover, deforestation, urban expansion, agriculture, and vegetation health. **Precision Agriculture:** Optimization of yield, resource use, and sustainability through the analysis of geospatial data. **Water Resources:** Monitoring of water bodies, surface water extent, and detection of algal blooms. **Climate Studies:** Analysis of long-term trends in surface temperature and glacial changes. **Artificial Intelligence/Machine Learning:** Landsat data are widely used as a training and validation dataset for Deep Learning models in remote sensing, such as change detection, land use classification, and semantic segmentation.

## Integration
Landsat data are freely accessible through various platforms and tools:
*   **USGS EarthExplorer (EE):** Graphical interface to define areas of interest, select dates, and search multiple datasets simultaneously. Allows the download of Level 1, Level 2, and ARD products.
*   **Landsat in the Cloud (AWS):** Access to operational Landsat Collection 2 products on the Amazon Web Services (AWS) platform, using SpatioTemporal Asset Catalog (STAC) metadata.
*   **USGS ESPA (EROS Science Processing Architecture):** On-demand interface to request spectral indices (NDVI, SAVI, etc.) and higher-level products (Aquatic Reflectance, Actual Evapotranspiration). Supports Bulk API and Downloader for bulk data retrieval.
*   **M2M (Machine to Machine) API:** JSON RESTful API for programmatic access to the datasets, offering the same options as EarthExplorer for scripts.
*   **Other Tools:** Access and visualization are also possible through the Sentinel Hub EO Browser and Esri Landsat Explorer.

## URL
[https://www.usgs.gov/landsat-missions/data](https://www.usgs.gov/landsat-missions/data)
