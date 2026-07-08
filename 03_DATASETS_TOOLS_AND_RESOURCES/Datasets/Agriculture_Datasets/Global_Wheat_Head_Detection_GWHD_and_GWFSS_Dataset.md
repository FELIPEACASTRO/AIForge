# Global Wheat Head Detection (GWHD) Dataset and Global Wheat Full Semantic Organ Segmentation (GWFSS) Dataset

## Description

The **Global Wheat Head Detection (GWHD) Dataset** is one of the largest and most diverse datasets of labeled high-resolution wheat images, created by an international consortium of 11 institutions across 7 countries. The main goal is to develop and compare computer vision methods for detecting and counting wheat heads under field conditions.

The original version (GWHD 2020) was expanded into **GWHD 2021**, which improved the diversity and reliability of the labels. More recently, the 2025 paper introduced the **Global Wheat Full Semantic Organ Segmentation (GWFSS) dataset**, an evolution aimed at complete semantic segmentation of plant organs (leaves, stems, and heads), covering all developmental stages, which represents a significant advance over the bounding-box detection of GWHD.

The dataset is crucial for advancing high-throughput phenotyping, enabling the estimation of important traits such as head population density, a key component of wheat yield. The diversity of genotypes, environments, and image acquisition conditions makes GWHD a robust benchmark for deep learning models.

## Statistics

- **GWHD 2020 (Original):** 4,700 high-resolution RGB images and approximately 190,000 labeled wheat heads.
- **GWHD 2021 (Update):** 6,515 high-resolution RGB images, representing 275,187 labeled wheat heads.
- **GWFSS (2025 - Evolution):** 1,096 diverse images with pixel-level semantic organ segmentation (leaves, stems, and heads) and an additional set of 52,078 unannotated images for pre-training.
- **File Size (Kaggle):** The Kaggle competition version has a size of 643.57 MB (3,434 files, including images and CSVs).
- **Label Density:** An average of 40 heads per image, with a wide distribution.
- **Resolution:** Images harmonized to an effective resolution of 0.21 to 0.55 mm/pixel.

## Features

- **Global Diversity**: Images collected across 7 countries (Japan, France, Canada, United Kingdom, Switzerland, China, Australia) and 10 locations, covering a wide range of genotypes, environmental conditions, and developmental stages.
- **High Resolution**: High-resolution RGB images (originally up to 6000x4000 pixels) that were harmonized and split into patches of 1024x1024 pixels.
- **Bounding Box Labels**: The original GWHD dataset provides bounding boxes for each identified wheat head.
- **Evolution to Semantic Segmentation (GWFSS)**: The most recent evolution (GWFSS, 2025) provides pixel-level segmentation for complete organs (leaves, stems, and heads), including necrotic and senescent tissue, which is fundamental for classifying healthy vs. unhealthy tissue.
- **Generalization Challenge**: The dataset was intentionally split to test generalization, with the training set focused on Europe/North America and the test set on Australia/Japan/China.

## Use Cases

- **Wheat Head Detection and Counting:** The primary use case, crucial for estimating head population density, a key component of yield.
- **High-Throughput Phenotyping:** Extraction of additional phenotypic traits, such as size, inclination, color, maturity stage, and head health.
- **Computer Vision Model Development:** A benchmark for the development and evaluation of deep learning models (e.g., YOLO, Faster R-CNN, Segformer) on object detection and semantic segmentation tasks in complex agricultural environments.
- **Model Generalization Study:** The dataset is ideal for testing the robustness of models under unseen conditions (different genotypes, environments, and cameras).
- **Semantic Organ Segmentation (GWFSS):** Advanced application for classifying healthy vs. unhealthy tissue, quantifying senescence and diseases in wheat canopies.

## Integration

The dataset is publicly accessible and was the basis for the Kaggle "Global Wheat Detection" competitions (2020 and 2021).

**Dataset Access:**
1.  **Official Website:** The dataset is available for download and further information at [http://www.global-wheat.com/](http://www.global-wheat.com/).
2.  **Kaggle:** The original competition data is available on Kaggle, although the competition itself is closed: [https://www.kaggle.com/competitions/global-wheat-detection/data](https://www.kaggle.com/competitions/global-wheat-detection/data).

**Integration Example (Conceptual - Python/PyTorch/TensorFlow):**
Integration typically involves downloading the image files (`train.zip`, `test.zip`) and the label file (`train.csv`). The `train.csv` contains the bounding boxes in the format `[xmin, ymin, width, height]`.

```python
import pandas as pd
import os

# Load the label file
labels_df = pd.read_csv('train.csv')

# Example DataFrame structure
# image_id, width, height, bbox
# 8c148c04c, 1024, 1024, [10, 20, 50, 60]

def parse_bbox(bbox_str):
    # Converts the bbox string to a list of integers
    # Ex: "[10, 20, 50, 60]" -> [10, 20, 50, 60]
    import ast
    return ast.literal_eval(bbox_str)

# Apply the function to obtain the coordinates
labels_df['bbox_coords'] = labels_df['bbox'].apply(parse_bbox)

# Example of how to load an image and its bounding boxes
image_id = labels_df['image_id'].iloc[0]
image_path = os.path.join('train', f'{image_id}.jpg')

# In a Deep Learning pipeline (e.g., with PyTorch or TensorFlow),
# the next step would be to load the image, apply transformations,
# and format the labels into the format expected by the model (e.g., YOLO, Faster R-CNN).
# The bbox coordinates would be used to train the object detection model.
```

## URL

http://www.global-wheat.com/
