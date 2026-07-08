# DeepWeeds Dataset

## Description

DeepWeeds is a public, annotated multiclass image dataset designed for the robust recognition of weed species in rangeland environments in Australia. It was created to address the need for realistic field data for the development of robotic weed control systems. The dataset is widely used in deep learning research for image classification in precision agriculture.

## Statistics

**Total Images:** 17,509 unique color images.
**Image Resolution:** 256x256 pixels.
**Standard Split:**
*   **Training:** 60% (approximately 10,505 images)
*   **Validation:** 20% (approximately 3,502 images)
*   **Test:** 20% (approximately 3,502 images)
**Classes:** 9 classes (8 weed species and 1 background/other plants class).
**Balancing:** The dataset was collected with the goal of having at least 1,000 images per species and a balanced split between positive samples (weeds) and negative samples (background/other plants) per site.

## Features

**Data Type:** Color (RGB) images of 256x256 pixels.
**Classes:** 9 classes in total (8 weed species and 1 "Other" plants/background class).
**Weed Species:** Chinee Apple (_Ziziphus mauritiana_), Lantana (_Lantana camara_), Parkinsonia (_Parkinsonia aculeata_), Parthenium (_Parthenium hysterophorus_), Prickly Acacia (_Vachellia nilotica_), Rubber Vine (_Cryptostegia grandiflora_), Siam Weed (_Chromolaena odorata_), and Snake Weed (_Stachytarpheta spp._).
**Collection:** Images collected _in situ_ (on site) across eight rangeland environments in northern Australia, reflecting variable environmental conditions (lighting, occlusion, dynamic background).
**Resolution:** Approximately 4 pixels per mm.

## Use Cases

*   **Weed Species Classification:** Training and evaluation of Deep Learning models (such as CNNs, ResNet, Inception) to automatically identify the weed species present in an image.
*   **Precision Agriculture:** Development of computer vision systems for agricultural robots or drones, enabling localized and selective herbicide application (site-specific weed control).
*   **Computer Vision Research:** Study of _feature engineering_ techniques and neural network architectures to handle background variability, occlusion, and lighting conditions in field environments.
*   **Transfer Learning:** Use of the dataset to fine-tune models pre-trained on large datasets (such as ImageNet) for the specific task of weed classification.

## Integration

The DeepWeeds dataset is available on platforms such as Kaggle and TensorFlow Datasets (TFDS), facilitating access and integration.

**Access Example via Kaggle API (Shell):**
```bash
# Install the Kaggle CLI (if needed)
# pip install kaggle

# Download the dataset (requires configured Kaggle credentials)
kaggle datasets download -d imsparsh/deepweeds
unzip deepweeds.zip
```

**Loading Example in Python (TensorFlow Datasets):**
```python
import tensorflow_datasets as tfds

# Load the dataset
(ds_train, ds_test), ds_info = tfds.load(
    'deep_weeds',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Print information
print(ds_info)

# Iteration example
for image, label in ds_train.take(1):
    print(f"Image Shape: {image.shape}")
    print(f"Label: {label.numpy()}")
```

## URL

https://www.nature.com/articles/s41598-018-38343-3
