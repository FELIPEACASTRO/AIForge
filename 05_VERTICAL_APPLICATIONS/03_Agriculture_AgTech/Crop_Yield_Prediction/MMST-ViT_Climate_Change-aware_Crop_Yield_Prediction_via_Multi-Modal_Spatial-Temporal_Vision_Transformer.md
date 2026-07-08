# MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer

## Description

**MMST-ViT** (Multi-Modal Spatial-Temporal Vision Transformer) is a deep learning solution developed for **crop yield prediction** at the county level in the United States. The model is notable for its ability to integrate and process multimodal, spatial-temporal data, addressing the challenges posed by short-term meteorological variations and long-term climate change on crop growth.

The MMST-ViT architecture is composed of three main components:
1.  **Multi-Modal Transformer:** Combines visual remote sensing data (Sentinel-2 imagery) with short-term meteorological data (WRF-HRRR) to model the direct impact of seasonal climate conditions.
2.  **Spatial Transformer:** Learns the high-resolution spatial dependency between counties, enabling precise agricultural tracking.
3.  **Temporal Transformer:** Captures long-range temporal dependency, essential for modeling the impact of long-term climate change on crops.

The model also uses a multimodal contrastive learning technique for pre-training, reducing the need for extensive human supervision.

## Statistics

MMST-ViT demonstrated superior performance compared to its counterparts in extensive experiments across more than 200 U.S. counties.

**Performance Metrics (Soybean Prediction):**
- **Root Mean Square Error (RMSE):** **3.9** (the lowest among the compared models)
- **R-squared ($R^2$):** **0.843** (the highest among the compared models)
- **Correlation (Corr):** **0.918**

**Citations:** The article was published at the **IEEE/CVF International Conference on Computer Vision (ICCV) in 2023** and has a significant number of citations (more than 76, according to arXiv), indicating its relevance in the research community.

**Dataset:** Uses **Tiny CropNet** and **CropNet**, which span data from 2017 to 2022 across more than 200 U.S. counties.

## Features

- **Hybrid Transformer Architecture:** Combines Multi-Modal, Spatial, and Temporal Transformers.
- **Multimodality:** Integrates satellite imagery (Sentinel-2), meteorological data (WRF-HRRR), and crop data (USDA Crop Dataset).
- **Climate Awareness:** Designed to capture the impacts of short- and long-term climate variations.
- **Contrastive Pre-training:** Uses multimodal contrastive learning for efficient pre-training.
- **County-Level Prediction:** Offers highly granular predictions for large geographic areas.

## Use Cases

- **Crop Yield Prediction:** Primary application, focused on crops such as soybean, corn, and cotton at the county level.
- **Precision Agricultural Monitoring:** The Spatial Transformer enables precise, high-resolution agricultural tracking.
- **Climate Impact Analysis:** Assessment of the effects of short-term meteorological variations and long-term climate change on agricultural production.
- **Agricultural Planning and Decision-Making:** Provides valuable information to optimize production and ensure food supply.

## Integration

The official implementation of MMST-ViT is available on GitHub, providing the source code in PyTorch and detailed instructions for pre-training and fine-tuning.

**Main Requirements:**
- `torch == 1.13.0`
- `torchvision == 0.14.0`
- `timm == 0.5.4`
- `numpy == 1.24.4`
- `pandas == 2.0.3`

**Usage Example (Fine-Tuning):**
The fine-tuning process for crop yield prediction can be started with the following command in the PyTorch environment:

```python
# Install dependencies
pip install -r requirements.txt

# Generate JSON configuration file for the dataloader (example for soybean)
python config/build_config_soybean.py

# Run fine-tuning
python main_finetune_mmst_vit.py
```

The repository also provides the **Tiny CropNet** dataset and **CropNet** (an extension) on HuggingFace Datasets, facilitating reproduction and use.

## URL

https://github.com/fudong03/MMST-ViT