# Air Quality Datasets (EPA) - Air Quality System (AQS)

## Description
The U.S. Environmental Protection Agency (EPA) Air Quality dataset, known as the Air Quality System (AQS), is the official repository of ambient air quality data collected at more than 10,000 monitors across the United States, Puerto Rico, and the U.S. Virgin Islands. AQS stores measurements of atmospheric pollutants, including the six criteria air pollutants (ozone, particulate matter (PM2.5 and PM10), carbon monoxide, sulfur dioxide, nitrogen dioxide, and lead), as well as toxic air pollutant and meteorological data. The data are crucial for air quality assessment, pollutant dispersion modeling, and compliance reporting with the National Ambient Air Quality Standards (NAAQS).

## Statistics
**Monitors:** More than 10,000 monitors in total, with approximately 5,000 active. **Time Series:** Data available since 1990. **Samples (Annual Example):** The 2023 annual concentration by monitor file contains 81,007 rows of data (4,132 KB compressed). The monitor description file contains 367,437 rows (6,701 KB compressed). **Updates:** The pre-generated files are updated twice a year (June and December), with 2025 data already partially available (through July 31, 2025).

## Features
**Geographic Scope:** United States, Puerto Rico, and the U.S. Virgin Islands. **Pollutants:** Includes the six criteria pollutants (O3, PM2.5, PM10, CO, SO2, NO2, Pb), as well as toxic air pollutants and meteorological data. **Granularity:** Data available at different aggregation levels: raw samples, hourly, daily, and annual summaries. **Formats:** Pre-generated and compressed CSV (comma separated variable) files (.zip). **Metadata:** Includes detailed descriptions of sites and monitors.

## Use Cases
**AI Modeling:** Training of Machine Learning (ML) and Deep Learning (DL) models for air quality prediction (ozone, PM2.5, etc.), trend identification, and time-series analysis. **Public Health Research:** Study of the correlation between air pollution and health outcomes, such as respiratory and cardiovascular diseases. **Policy Evaluation:** Assessment of the effectiveness of environmental regulations and the National Ambient Air Quality Standards (NAAQS). **Exposure Modeling:** Creation of high-resolution air pollution exposure models for specific communities.

## Integration
The dataset is primarily accessible through the EPA's AirData page, which offers: 1. **Pre-Generated Files:** ZIP files containing data in CSV format, grouped by year and aggregation type (annual, daily, hourly). The files are updated twice a year (June and December). 2. **Daily Download Tool:** An interactive tool to query and download daily data for criteria pollutants. 3. **API:** The data are also available via API for programmatic integration, although the API documentation should be consulted separately on the EPA website. Users should be familiar with the EPA's air quality monitoring program to correctly interpret the data.

## URL
[https://aqs.epa.gov/aqsweb/airdata/download_files.html](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
