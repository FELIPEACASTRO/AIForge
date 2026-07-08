# RicEns-Net (Deep Ensemble Model)

## Description

RicEns-Net is an innovative Deep Ensemble model designed to improve the accuracy of agricultural yield prediction. It uses multimodal data fusion techniques to integrate diverse sources of information, such as optical remote sensing data and synthetic aperture radar (SAR) data from the Sentinel 1, 2, and 3 satellites, as well as meteorological measurements including surface temperature and precipitation. The model was developed from data of Ernst & Young's (EY) Open Science Challenge 2023. A data engineering step reduced more than 100 potential predictors to an optimized set of 15 features from 5 distinct modalities, mitigating the "curse of dimensionality" and improving performance. Its architecture combines multiple machine learning algorithms within a deep ensemble framework to maximize predictive accuracy.

## Statistics

- **Mean Absolute Error (MAE):** 341 kg/Ha.
- **Performance Improvement:** Significantly exceeds the performance of previous state-of-the-art models, including those developed in the EY challenge. The MAE of 341 kg/Ha corresponds to approximately 5-6% of the lowest average yield in the studied region.
- **Publication:** Submitted on February 9, 2025.
- **Citations:** 10 (in 2025, according to the initial snippet).
- **Training Data:** Data from Ernst & Young's (EY) Open Science Challenge 2023.

## Features

- **Multimodal Data Fusion:** Integration of SAR (Sentinel 1), optical (Sentinel 2 and 3), and meteorological (temperature and precipitation) data.
- **Deep Ensemble Learning:** Combination of multiple machine learning algorithms in a deep ensemble architecture.
- **Optimized Feature Engineering:** Selection of the 15 most informative features from more than 100 potential predictors.
- **High Accuracy:** Performance superior to previous state-of-the-art models.

## Use Cases

- **Agricultural Yield Prediction:** Accurate crop yield prediction, with an initial focus on rice crops (implied by the name RicEns-Net, although the abstract is generic).
- **Precision Agriculture:** Decision-making support for farmers and agricultural managers, optimizing the use of resources (fertilizers, irrigation).
- **Risk Assessment:** Use by insurers and financial institutions to assess crop risk across different regions.
- **Regional Monitoring:** Application over large geographic areas, using satellite data for monitoring and prediction at a regional scale.

## Integration

The article is academic research and does not provide a ready-to-use code integration guide. However, the methodology suggests implementation in a machine learning framework (such as PyTorch or TensorFlow) that supports:
1.  **Data Preprocessing:** Normalization and alignment of multimodal data (SAR, optical, meteorological).
2.  **Feature Selection:** Use of techniques such as *feature importance* to reduce the set of predictors.
3.  **Ensemble Architecture:** Implementation of a deep ensemble structure where the predictions of base models (such as CNNs, LSTMs, or traditional ML models) are combined by a meta-learner.
The source code and dataset may be available in the repository associated with the article (not explicitly listed in the abstract).

## URL

https://arxiv.org/abs/2502.06062