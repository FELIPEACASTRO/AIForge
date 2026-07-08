# AGBD: A Global-scale Biomass Dataset

## Description

The **AGBD (A Global-scale Biomass Dataset)** is a new global reference dataset of aboveground biomass (AGB) designed for training machine learning models. It combines field AGB reference data with multimodal time-series remote sensing data from the GEDI, Sentinel-1 and PALSAR-2 satellites. The dataset is representative of diverse vegetation typologies and spans several years, making it ideal for models that exploit the temporal dimension. The goal is to provide a high-resolution (10m) and globally representative benchmark for AGB estimation, overcoming the limitations of existing datasets that are too localized or of low resolution. The dataset is accompanied by benchmark models and is publicly available.

## Statistics

- **Coverage:** Global, spanning diverse vegetation typologies.
- **Resolution:** 10-meter AGB prediction map.
- **Input Data:** Combination of GEDI AGB data, Sentinel-1 and PALSAR-2 time series, canopy density map, elevation map, and land-use classification map.
- **Estimated Size:** The full dataset is large, with the paper mentioning that the AGB reference data is about 60 times larger than ImageNet, which would require approximately 70TB of storage. However, the version available for ML is a preprocessed and more accessible version.
- **Period:** Spans several years (time series).

## Features

- **Global and Representative:** Covers diverse vegetation typologies at global scale.
- **Multimodal and Time Series:** Combines GEDI AGB data with Sentinel-1 (radar) and PALSAR-2 (radar) time series, plus canopy density, elevation, and land-use classification maps.
- **High Resolution:** Provides a high-resolution (10m) AGB prediction map for the entire dataset coverage area.
- **Machine Learning Ready:** The dataset is preprocessed and compatible with frameworks such as TensorFlow and PyTorch.
- **Benchmark:** Includes rigorously tested benchmark models.

## Use Cases

- **AGB Model Training:** Used to train machine learning (ML) models for Aboveground Biomass (AGB) estimation at global scale.
- **Benchmark:** Serves as a baseline for validating the accuracy and reliability of new AGB estimation models.
- **Regional Performance Improvement:** Allows researchers to fine-tune global models with local reference data to improve regional accuracy and performance.
- **Carbon and Biodiversity Studies:** Supports the assessment of forest carbon stocks and biodiversity structure.

## Integration

The dataset is hosted on HuggingFace (Lhoest et al., 2021) and can be downloaded and used with the following lines of code (Python):

```python
# Installation of the datasets library
!pip install datasets

# Loading the AGBD dataset
from datasets import load_dataset
dataset = load_dataset("pre-eth/AGBD", streaming=True)["train"] # or test, val
```

## URL

https://isprs-annals.copernicus.org/articles/X-G-2025/829/2025/isprs-annals-X-G-2025-829-2025.pdf