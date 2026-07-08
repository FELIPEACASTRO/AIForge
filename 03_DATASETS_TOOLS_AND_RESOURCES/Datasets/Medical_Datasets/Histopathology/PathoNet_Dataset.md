# PathoNet Dataset

## Description

**PathoNet** is a general-purpose dataset for digital pathology, focused on histopathology images. It was published in July 2023 and consists of images extracted from the **TCGA (The Cancer Genome Atlas)** data portal. The dataset is composed of images from 12 different tissue classes, making it a valuable resource for the development and evaluation of *Deep Learning* models in computational pathology. The images are 256x256 pixel patches extracted from Whole Slide Images (WSI), and they went through an automated cleaning process to remove excessive white content and blurred images. The split into Training, Validation, and Test sets is done by considering cases (WSI) to avoid mixing images from the same case across different partitions.

## Statistics

**Total Images:** 4,462,156 JPG images.
**Resolution:** 256x256 pixels.
**Classes:** 12 tissue classes.
**Source:** TCGA (The Cancer Genome Atlas).
**Total File Size:** 131.6 GB (split into multiple .zip files).
**Data Split (Example - Bladder):** Training (308,677), Validation (38,927), Test (39,166).

## Features

General-purpose dataset for digital pathology. Histopathology images (256x256 pixel patches). 12 tissue classes (Bladder, Brain, Breast, Bronchus and Lung, Colon, Corpus Uteri, Kidney, Liver and Intrahepatic Bile Ducts, Prostate, Skin, Stomach, Thyroid Gland). Extracted from Whole Slide Images (WSI) of TCGA. Case-based data split to avoid data leakage. Automated cleaning process.

## Use Cases

Training and evaluation of Deep Learning models for tissue classification in digital pathology. Development of segmentation and object detection models on histopathology images. Research in computational pathology, especially for multi-class tissue-based cancer classification tasks. Transfer Learning studies in pathology.

## Integration

The dataset is available for download on Zenodo, split into `.zip` files by partition (Training, Validation, Test) and by class for the training set. The download can be performed directly through the Zenodo web interface.

**Download Example (Shell/cURL):**
Due to the total size of 131.6 GB, the download should be done in parts. The `test.zip` file (13.3 GB) can be downloaded via `cURL` or `wget` using the direct download link provided on the Zenodo page.

```bash
# Example of downloading the test file
wget https://zenodo.org/record/8116751/files/test.zip
# Example of downloading a training class (Bladder)
wget https://zenodo.org/record/8116751/files/train_bladder.zip
```

**Note:** The direct download URL may change, with the Zenodo page being the primary source for obtaining the most recent links. There is no standard Python integration script provided, but the download and handling of the `.zip` files can be easily integrated into an ML pipeline using libraries such as `zipfile` and `Pillow` in Python.

## URL

https://zenodo.org/records/8116751