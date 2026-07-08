# DeepBioFusion (Model), BioMassters (Dataset), DeepLiDARPlanet (Implementation)

## Description

Comprehensive research on the use of LiDAR and deep learning techniques for Above Ground Biomass (AGB) estimation, focusing on recent developments (2023-2025). Three key resources were identified: a multimodal deep learning model (DeepBioFusion), a benchmark dataset (BioMassters), and a code repository for implementation (DeepLiDARPlanet). The results detail the characteristics, performance statistics, use cases, and integration information for each resource.

## Statistics

- **DeepBioFusion:** Correlation coefficient of 0.57 (SAR L-band) between estimated and predicted biomass values. Local prediction errors (AGB) within a range of ±5.1%.
- **BioMassters:** Dataset released in 2023, cited 17 times (NeurIPS 2023). Contains 5 years of data (2016-2021) for nearly 13,000 forest patches in Finland.
- **DeepLiDARPlanet:** Repository with 7 stars and 1 fork (as of 2025). Focuses on U-Net models for height and biomass prediction at 100m spatial resolution.

## Features

- **DeepBioFusion:** Multimodal deep learning framework (SAR and optical) for tree-level AGB estimation. Uses LiDAR-derived data for ground truth generation and incorporates species modeling.
- **BioMassters:** Benchmark dataset with multimodal time series (Sentinel-1 SAR and Sentinel-2 MSI) for AGB estimation in Finnish forests.
- **DeepLiDARPlanet:** Repository with tools and models for forest height and biomass estimation, integrating LiDAR and Planet data, with a focus on U-Net models for prediction at 100m resolution.

## Use Cases

- **DeepBioFusion:** Large-scale biomass monitoring, climate change mitigation, sustainable forest management, and wildfire risk assessment.
- **BioMassters:** Training and benchmarking of deep learning models for AGB estimation using multimodal satellite data.
- **DeepLiDARPlanet:** Creation of continuous vegetation maps for forest management, reforestation/deforestation monitoring, and land cover classification in tropical regions (e.g., Kalimantan).

## Integration

- **DeepBioFusion:** There is no direct integration code available in the article, but the framework is based on deep learning and uses satellite data (SAR and optical) and reference data derived from LiDAR.
- **BioMassters:** The dataset is available on Hugging Face and the competition was hosted on DrivenData. The baseline code is accessible on GitHub (link in the URL).
- **DeepLiDARPlanet:** The GitHub repository provides the folder structure and Python scripts for data preprocessing (CHM, Planet) and modeling (U-Net, Random Sampling), with installation requirements (Python 3.11.4, PyTorch 11.7, Rasterio, GDAL). The integration code is based on Python/PyTorch.

## URL

DeepBioFusion: https://www.sciencedirect.com/science/article/pii/S1574954125002869 | BioMassters: https://nascetti-a.github.io/BioMasster/ | DeepLiDARPlanet: https://github.com/fraware/LiDAR-Planet-Deep-Learning