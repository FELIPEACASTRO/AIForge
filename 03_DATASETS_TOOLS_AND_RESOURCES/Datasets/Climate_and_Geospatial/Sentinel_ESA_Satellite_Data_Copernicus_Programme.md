# Sentinel (ESA Satellite Data) - Copernicus Programme

## Description
The Sentinel programme is the Earth observation component of the European Union's Copernicus programme, managed by the European Space Agency (ESA). It consists of a constellation of satellites (Sentinel-1, -2, -3, -5P, etc.) that provide Earth observation data continuously and free of charge. The data covers a wide range of applications, from land and ocean monitoring to air quality and emergency management. The main focus is to provide data for the operational Copernicus services, ensuring global coverage and a high revisit time.

## Statistics
The total Copernicus Sentinel data archive is massive, with the Copernicus Data Space Ecosystem (CDSE) reporting a total of **20.8 Petabytes (PB)** of products available as of January 2024. The data volume grows at a rate of **Terabytes per day**. For example, a Sentinel-2 Level-1B granule is approximately 27 MB. The Sentinel-1 programme publishes more than 95,000 products monthly. **Versions:** The programme is continuous, with satellite launches and replacements (e.g., Sentinel-2C launched in September 2024, replacing Sentinel-2A in January 2025).

## Features
**Mission Constellation:** Includes radar missions (Sentinel-1), multispectral optical missions (Sentinel-2), ocean and land monitoring (Sentinel-3), and atmospheric monitoring (Sentinel-5P). **High Resolution and Revisit:** Sentinel-2, for example, offers 13 spectral bands at resolutions of 10m, 20m, and 60m, and a revisit time of 5 days at the Equator with two satellites. **Free and Open Access:** All data is provided free of charge and openly to global users. **Processing Levels:** Products range from Level-1C (Top-of-Atmosphere Reflectance) to Level-2A (Atmospherically Corrected Surface Reflectance), the latter being compatible with CEOS Analysis Ready Data (CEOS-ARD).

## Use Cases
**Land Monitoring:** Land use and land cover mapping, precision agriculture, and forest monitoring (forestry). **Marine Monitoring:** Observation of oceans, sea ice, pollution, and water colour. **Emergency Management:** Mapping of floods, wildfires, and natural disasters. **Climate and Atmosphere:** Monitoring of greenhouse gases and air quality (Sentinel-5P). **Security:** Security and maritime surveillance applications.

## Integration
Data access is provided mainly through the **Copernicus Data Space Ecosystem (CDSE)**, which replaced the Copernicus Open Access Hub. CDSE offers several access options: **Direct Download:** Through the CDSE web portal. **APIs:** Use of RESTful APIs, such as the Sentinel Hub API and the Streamlined Data Access APIs (SDA), for programmatic access and product download. **Processing Platforms:** Integration with cloud analysis platforms such as JupyterLab, openEO, and Sentinel Hub, allowing data processing without the need for local download of large volumes. **Software:** Tools such as ESA's SNAP (Sentinel Application Platform) are recommended for data processing and visualization.

## URL
[https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
