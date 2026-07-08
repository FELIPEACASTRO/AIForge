# AgriTransformer (Co-attention variant)

## Description

AgriTransformer is a *deep learning* architecture based on the *Transformer* that uses a **co-attention (cross-modal attention)** mechanism to integrate multimodal data (tabular agricultural management data and vegetation indices from satellite imagery) in order to improve crop yield prediction. The co-attention mechanism allows the model to learn the dynamic interdependencies between the different data modalities, outperforming models that treat the modalities separately or simply concatenate them. The model was developed to capture the complex nonlinear interactions between environmental conditions, management practices, and the physiological responses of plants.

## Statistics

- **Best Performance:** Variants with co-attention.
- **Metrics:** MSE = 2.598 (Mean), R² = 0.919 (Mean).
- **Comparison:** Significantly outperforms Linear Regression (R² = 0.704) and Dense Neural Networks (R² = 0.884).
- **Dataset:** Telangana Crop Health Challenge (India).
- **Citations:** 3 (as of the article's publication date in 2025).

## Features

- **Co-Attention Mechanism (Cross-Modal Attention):** Enables the efficient fusion of heterogeneous data (tabular and spectral), learning how the features of one modality influence the other.
- **Transformer-Based Architecture:** Leverages the strength of *Transformers* to model complex, long-range dependencies in the data.
- **Multimodal Inputs:** Simultaneously processes tabular data (crop type, irrigation, management) and vegetation indices (NDVI, EVI, NDWI, GNDVI, SAVI, MSAVI) derived from Sentinel-2 satellite imagery.
- **Modularity:** Modular design that facilitates adaptability and transferability to different crop types and geographic regions.

## Use Cases

- **Crop Yield Prediction:** Accurate estimation of agricultural production at the field level, crucial for food security and logistics planning.
- **Precision Agriculture:** Support for real-time decision-making on irrigation, fertilization, and crop management.
- **Modeling Complex Interactions:** Identification and quantification of the influence of management factors (irrigation, crop type) on vegetation health conditions (VIs).
- **Agricultural Monitoring:** Provides a robust foundation for monitoring crop health and development throughout the growing season.

## Integration

The model was implemented using the **TensorFlow** and **Keras** (Python) *frameworks*, with **Google Earth Engine** being used to obtain the multispectral satellite imagery. The integration involves:
1.  **Data Preprocessing:** Normalization of tabular data and calculation of Vegetation Indices (VIs) from Sentinel-2 imagery.
2.  **Architecture:** Construction of the *Embedding* layers for tabular data and VIs, followed by the Co-Attention module.
3.  **Training:** Adam optimizer, initial *learning rate* of 0.001, MSE loss function.
*No public GitHub repository with the implementation code was found at the time of research, but the article details the architecture for replication.*

## URL

https://www.mdpi.com/2079-9292/14/12/2466