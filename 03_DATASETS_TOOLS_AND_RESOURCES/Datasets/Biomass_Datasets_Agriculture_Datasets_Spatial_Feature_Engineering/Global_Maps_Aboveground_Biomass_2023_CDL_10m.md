# Global Maps of Aboveground Biomass, Canopy Height and Cover for 2023, Cropland Data Layer (CDL) 2024 - 10m Resolution, Geospatial Feature Engineering Techniques (GLCM and CAE-based Features)

## Description

Global dataset of aboveground biomass density (AGBD), canopy height (CH) and canopy cover (CC) at 10-meter resolution, derived from a Unified Deep Learning Model applied to high-resolution multi-sensor satellite imagery. The data are accompanied by standard error maps for each variable. The Cropland Data Layer (CDL) is a georeferenced, crop-specific raster land cover map produced annually by the USDA National Agricultural Statistics Service (NASS). The 2024 version marks the transition of spatial resolution from 30 meters to 10 meters, providing unprecedented field-level detail. Advanced spatial feature engineering techniques, such as the Gray-Level Co-occurrence Matrix (GLCM) for texture extraction and Convolutional Autoencoder (CAE)-based features for latent feature extraction, are crucial for improving accuracy in Biomass and Agriculture models.

## Statistics

Year: 2023. Resolution: 10 meters. Coverage: Global (57° S to 67° N). Total Size: 5 TB (Complete Dataset). Variables (6 GeoTIFF bands): AGBD (Mg/ha), CH (cm), CC (%), AGBD Standard Error, CH Standard Error, CC Standard Error. Year: 2024 (Released Feb/2025). Resolution: 10 meters (New resolution) and 30 meters (resampled version). Coverage: United States (Continental and Hawaii). Classes: More than 110 crop categories and 14 non-agricultural categories. GLCM: Used in 2024 and 2025 studies for crop classification and organic carbon assessment. CAE-based Features: 2025 studies show up to 10% improvement in crop yield prediction compared to traditional autoencoders.

## Features

High resolution (10m); Multi-sensor fusion (Deep Learning Model); Includes uncertainty estimates (Standard Error); Global coverage for 2023. High resolution (10m) for field-level detail; Crop-specific (identification of crop types); Basis for the Agricultural Cropland Tracking and Interactive Visualization Environment (ACTIVE) on Google Earth Engine; Historical consistency with 30m data. GLCM (Gray-Level Co-occurrence Matrix): Extracts texture features (Contrast, Homogeneity, Entropy, Correlation) capturing the spatial relationship between pixels. CAE (Convolutional Autoencoder) Latent Features: Learns compact representations of high-dimensional satellite data, separating genotype and environment features.

## Use Cases

Climate change mitigation and carbon accounting (AGBD); Forest management and inventory (CH, CC); Ecological modeling and biodiversity studies; Agricultural monitoring (Canopy Cover). Assessment of planted area and agricultural production; Monitoring of land cover changes; Crop yield modeling; Analysis of agricultural and environmental policies. Crop classification and land-use mapping (GLCM); Biomass and organic carbon estimation (GLCM); Early-stage crop yield prediction (CAE); Plant disease detection and classification (CAE).

## Integration

Format: GeoTIFF files (3° x 3° tiles). Access: Zenodo (48 GB subsample) or AWS S3 Requester-Pays service (complete dataset). AWS S3 Command: aws s3 sync s3://eda-appsci-open-access/biomass/ DESTINATION_PATH --request-payer requester. Access: Available in the CroplandCROS web application. Download: 30m version available for download. The 10m version can be accessed via CroplandCROS and possibly via Google Earth Engine (ACTIVE). GLCM: Implemented in libraries such as `scikit-image` (Python) or Google Earth Engine. CAE-based Features: Requires implementation and training of Convolutional Autoencoder models using Deep Learning frameworks (TensorFlow/PyTorch).

## URL

https://zenodo.org/records/15269923, https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php, References from scientific articles (e.g., MDPI, Frontiers in Plant Science, IEEE Xplore) from 2024-2025.