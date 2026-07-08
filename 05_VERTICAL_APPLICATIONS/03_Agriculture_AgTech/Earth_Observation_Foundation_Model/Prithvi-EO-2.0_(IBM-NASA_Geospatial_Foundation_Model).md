# Prithvi-EO-2.0 (IBM-NASA Geospatial Foundation Model)

## Description

Prithvi-EO-2.0 is a second-generation geospatial foundation model, jointly developed by IBM, NASA, and the Jülich Supercomputing Centre. It is a Transformer-based Vision (ViT) model, pre-trained using a masked autoencoder (MAE) approach. It was trained on 4.2M global time-series samples from NASA's Harmonized Landsat and Sentinel-2 (HLS) archive at 30m resolution. The model incorporates temporal and location embeddings for improved performance across various geospatial tasks. It is made available in different parameter sizes (300M and 600M), with and without time/location (TL) embeddings.

## Statistics

**Performance:** The 600M version outperforms the previous Prithvi-EO model by 8% on GEO-Bench. It outperforms six other geospatial foundation models on remote sensing tasks. **Downloads:** 54,018 downloads in the last month (data from 2024). **Citation:** Technical paper (arXiv:2412.02732) published in 2024, with more than 60 citations.

## Features

ViT architecture with 3D modifications for spatial-temporal data. Support for temporal and location embeddings. Pre-training with 30m HLS data (Landsat and Sentinel-2). Available in 300M and 600M versions, with and without TL (Temporal/Location embeddings). Apache-2.0 license. Uses the IBM TerraTorch framework for fine-tuning.

## Use Cases

Land use and crop mapping. Ecosystem dynamics monitoring. High-resolution applications (0.1m to 15m). Disaster response. Carbon Flux Prediction (Regression). Landslide segmentation. Multitemporal crop segmentation.

## Integration

The model can be used for fine-tuning with IBM's TerraTorch framework. To build the backbone in a custom PyTorch pipeline:

```python
from terratorch.registry import BACKBONE_REGISTRY

# Example for the tiny version with time/location embeddings
model = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True)
```

Example inference script for image reconstruction:

```bash
python inference.py --data_files t1.tif t2.tif t3.tif t4.tif --input_indices <optional, space separated 0-based indices of the six Prithvi channels in your input>
```

Example notebooks for fine-tuning are available on the project's GitHub.

## URL

https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M