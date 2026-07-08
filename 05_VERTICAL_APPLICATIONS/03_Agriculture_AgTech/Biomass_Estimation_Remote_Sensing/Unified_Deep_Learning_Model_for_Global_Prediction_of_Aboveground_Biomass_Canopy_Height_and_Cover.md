# Unified Deep Learning Model for Global Prediction of Aboveground Biomass, Canopy Height and Cover

## Description

A unified deep learning model that predicts Aboveground Biomass Density (AGBD), Canopy Height (CH) and Canopy Cover (CC), together with uncertainty estimates for all three quantities. It uses high-resolution (10 meter) multisensor, multispectral imagery and is trained on millions of globally sampled GEDI-L2/L4 measurements.

## Statistics

Trained on millions of globally sampled GEDI-L2/L4 measurements. Achieves a Mean Absolute Error (MAE) for AGBD (CH, CC) of 26.1 Mg/ha (3.7 m, 9.9%) and a Root Mean Square Error (RMSE) of 50.6 Mg/ha (5.4 m, 15.8%) on a globally sampled test dataset. Validated globally for 2023 and annually from 2016 to 2023 in selected areas. Demonstrates significant improvement over previously published results.

## Features

Unified prediction of AGBD, CH and CC. Provides uncertainty estimates. Uses high-resolution (10m) multisensor, multispectral satellite imagery. Multi-head architecture that facilitates transferability to other GEDI variables.

## Use Cases

Regular measurement of carbon stock in the world's forests. Carbon accounting and reporting under national and international climate initiatives. Scientific research on forest ecosystems.

## Integration

The abstract mentions a pre-trained model and a multi-head architecture that facilitates transferability, suggesting potential for reuse. The paper is from arXiv, so a direct link to the code or GitHub repository is not immediately available, but the model was deployed globally for 2023 by the authors.

## URL

https://arxiv.org/abs/2408.11234
