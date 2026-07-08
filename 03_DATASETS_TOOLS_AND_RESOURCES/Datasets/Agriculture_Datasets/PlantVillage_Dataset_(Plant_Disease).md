# PlantVillage Dataset (Plant Disease)

## Description

The PlantVillage Dataset is a collection of images of healthy and diseased plant leaves, and is one of the most widely used benchmarks for developing Deep Learning models for agricultural disease detection. The version most referenced and used in recent research (2023-2025) consists of 53,606 images of 14 crop species, divided into 38 classes (healthy and diseased). The dataset is crucial for advancing precision agriculture, enabling the training of fast and accurate diagnostic systems, including applications on edge computing devices. Recent research focuses on advanced feature extraction techniques and hybrid models to improve accuracy and reduce computational complexity.

## Statistics

**Total Images:** 53,606
**Image Dimensions:** 256 x 256
**Total Plant Species:** 14
**Total Classes (Healthy and Diseased):** 38

**Class Distribution (Examples):**
- **Tomato:** 10 classes (18,160 images)
- **Apple:** 4 classes (3,171 images)
- **Orange:** 1 class (5,507 images)

**Reference Accuracy (2025 Paper):** 98.95% (with a hybrid ResNet-KELM model).

## Features

Leaf images of 14 crop species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato). 38 distinct classes, including healthy states and 26 types of diseases. Images standardized to a dimension of 256x256 pixels. Used as a basis for Deep Feature Extraction techniques (ResNet-50, DenseNet, EfficientNet) and hybrid models (ResNet-KELM).

## Use Cases

**Disease Detection and Classification:** Identification of foliar diseases in crops.
**Mobile Application Development:** Creation of real-time diagnostic tools for farmers (edge computing).
**Research and Benchmarking:** Evaluation of new Deep Learning architectures and feature engineering techniques (e.g., ResNet-KELM, Feature Fusion, Attention Mechanisms).
**Transfer Learning:** Pre-training of models for disease classification tasks.

## Integration

The dataset is easily accessible via repositories such as Kaggle and TensorFlow Datasets (TFDS).

**Access Example (Python/TFDS):**
```python
import tensorflow_datasets as tfds

# Load the PlantVillage dataset
ds, ds_info = tfds.load('plant_village', split='train', with_info=True)

# Display dataset information
print(ds_info)

# The dataset is frequently used for Transfer Learning with pre-trained models.
```

## URL

https://www.kaggle.com/datasets/emmarex/plantdisease (Kaggle) or https://www.tensorflow.org/datasets/catalog/plant_village (TFDS)
