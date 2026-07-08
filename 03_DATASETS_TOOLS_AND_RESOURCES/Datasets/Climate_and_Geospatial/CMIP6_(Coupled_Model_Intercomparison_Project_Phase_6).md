# CMIP6 (Coupled Model Intercomparison Project Phase 6)

## Description
The **Coupled Model Intercomparison Project Phase 6 (CMIP6)** is an international project of the World Climate Research Programme (WCRP) that provides a coordinated set of global climate model (GCM) simulations to understand past, present, and future climate change. It is the primary source of data for the assessments of the Intergovernmental Panel on Climate Change (IPCC), including the Sixth Assessment Report (AR6). CMIP6 introduced a more complex experimental design, including the **DECK** experiments (Diagnosis, Evaluation, and Characterization of Klima) and a set of endorsed **MIPs** (Model Intercomparison Projects), which explore different aspects of the climate system and Shared Socioeconomic Pathways (SSPs). The data are essential for the global scientific community studying climate and its impacts.

## Statistics
- **Total Size:** Approximately **24.5 PB** (Petabytes) of data.
- **Dataset Count:** More than **6.4 million** individual datasets.
- **Versions/Models:** **322** experiments from **132** registered CMIP6 models.
- **Updates:** The project is ongoing, with data being continuously published and updated on the ESGF nodes. Recent versions (2023-2025) of derived datasets (e.g., downscaled) continue to be released.

## Features
- **Broad Coverage:** Includes 322 experiments from 132 registered global climate models (GCMs), originating from 48 scientific institutions in 26 countries.
- **Data Structure:** Output data are stored in **netCDF** files, with one variable per file, in compliance with the CF (Climate and Forecast) conventions and standardized by "controlled vocabularies" (CVs).
- **Future Scenarios:** Uses the **Shared Socio-economic Pathways (SSPs)** to project future climate under different scenarios of greenhouse gas emissions and socioeconomic development.
- **Improved Resolution:** CMIP6 models frequently feature higher spatial and temporal resolutions compared to previous phases (CMIP5).
- **Forcing Data:** Includes standard forcing data (e.g., greenhouse gas concentrations, aerosols) to ensure comparability between models.

## Use Cases
- **Climate Impact Assessment:** Primarily to provide climate projections for the IPCC and impact studies at global and regional scales.
- **Downscaling and Regionalization:** Creation of high-resolution datasets (e.g., 1 km) for regional studies of variables such as temperature and precipitation.
- **Water Resources Modeling:** Prediction of river discharge and drought assessment under future scenarios (SSPs).
- **Machine Learning (ML) Applications:** Use of CMIP6 data as input for ML models (e.g., Random Forest, Deep Learning) to improve the prediction of climate variables and their impacts in sectors such as agriculture.
- **Climate Extremes Studies:** Projection of changes in the frequency and intensity of extreme climate events.
- **Climate Sensitivity Analysis:** Investigation of the climate system's response to different forcings (e.g., aerosols, greenhouse gases).

## Integration
Access to CMIP6 data is managed by the **Earth System Grid Federation (ESGF)**, a distributed network of global data nodes (e.g., LLNL/USA, DKRZ/Germany, CEDA/United Kingdom, IPSL/France). Access is facilitated by the **Metagrid** web interface (replacing CoG).

**Download Methods:**
1.  **Wget Script:** Batch download scripts can be generated directly in the Metagrid interface.
2.  **Globus Transfer:** Recommended method for downloading large volumes of data, offering better performance. Requires an ESGF account and authentication via Globus Auth (supports Google, GitHub, and institutional accounts).
3.  **RESTful API:** Advanced users can use the ESGF Search RESTful API for programmatic access.

**Requirement:** It is necessary to create an ESGF account (using Globus Auth) for batch downloads and Globus transfers. Users must adhere to the CMIP6 Terms of Use, which require proper citation and acknowledgment of the data.

## URL
[https://wcrp-cmip.org/](https://wcrp-cmip.org/)
