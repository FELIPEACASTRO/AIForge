# Integration of Hyperspectral Imaging and AI for Crop Mapping and Health

## Description

Systematic review and analysis of recent trends (2023-2025) in the application of Hyperspectral Imaging (HSI) combined with advanced Artificial Intelligence (AI) models, such as Deep Learning (DL) and Vision Transformers (ViTs), for precision agriculture. The main focus is the accurate discrimination of crop types and the assessment of plant health, using data from airborne (UAV) and spaceborne (e.g., EnMAP, PRISMA) sensors. The research highlights the evolution toward more complex and scalable models, such as Graph Neural Networks (GNNs) and Geospatial Foundation Models (GFMs), as the next frontier for large-scale agricultural monitoring.

## Statistics

Recent research (2023-2025) points to the increasing use of Vision Transformers (ViTs) and hybrid architectures, which demonstrate high classification accuracy compared to traditional Machine Learning methods. The field is evolving rapidly, with 2025 review papers already citing the need to adopt Geospatial Foundation Models (GFMs) and Graph Neural Networks (GNNs) to handle the high dimensionality of HSI data and enable crop mapping over large areas. Specific examples in wheat include the use of Deep Learning with Attention Mechanism and Transfer strategies for yield estimation (2024), and Machine Learning for damaged grain analysis (2023).

## Features

Ability to accurately discriminate crop types; non-destructive and real-time assessment of biophysical parameters (e.g., nitrogen content, chlorophyll); early detection of water stress and diseases (e.g., *Fusarium head blight*); yield and biomass estimation; use of high-spectral-resolution data (hundreds of bands) from UAV and satellite platforms.

## Use Cases

Crop-type mapping at regional and global scales for food security; crop health monitoring and early-stage disease diagnosis; optimization of fertilizer (nitrogen) application and irrigation through nutrient and stress estimation; high-throughput phenotyping in genetic breeding programs; yield and biomass estimation for agricultural management and insurance.

## Integration

Integration is typically carried out through data processing pipelines that involve: 1. Preprocessing of HSI data (radiometric and atmospheric correction); 2. Dimensionality reduction (e.g., PCA, band selection); 3. Training of Deep Learning models (e.g., ViTs, CNNs) or Machine Learning models (e.g., SVM, Random Forest) for classification or regression. The GitHub repository `fadi-07/Awesome-Wheat-HSI-DeepLearning` serves as a reference resource for papers and implementations in wheat, indicating the basis for code development in Python with libraries such as PyTorch/TensorFlow for the DL models and scikit-learn for ML.

## URL

https://www.mdpi.com/2072-4292/17/9/1574
