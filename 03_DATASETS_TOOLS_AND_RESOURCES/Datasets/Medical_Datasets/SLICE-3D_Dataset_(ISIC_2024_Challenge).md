# SLICE-3D Dataset (ISIC 2024 Challenge)

## Description

The SLICE-3D Dataset is the official training dataset for the ISIC 2024 Challenge (International Skin Imaging Collaboration). It consists of skin lesion crops extracted from 3D total body photography (3D TBP), designed to mimic smartphone images submitted for telemedicine. The dataset targets the development of machine learning algorithms for skin cancer detection, focusing on standardized 15mm by 15mm lesions.

## Statistics

A total of 401,059 JPEG skin lesion images (CC-BY-NC version). The images are 15mm x 15mm crops. Clinical metadata includes age, sex, general anatomical site, patient identifier, clinical size, and malignancy values (gold standard).

## Features

The primary focus is on **Deep Learning (DL)** and **Radiomics**. Feature engineering techniques include: **Image feature extraction** (via CNNs and attention models), **Radiomic Features** (such as entropy and wavelet contrast), and **Hybrid Features** (combining image data and tabular clinical metadata such as age and sex). Advanced models use **two-stage segmentation and classification** frameworks.

## Use Cases

Development of **classification** models (malignant/benign) and **early skin cancer detection** in non-dermoscopic images. Applications in **telemedicine** and automated triage of skin lesions from cell phone photos.

## Integration

The dataset is available for download in compressed file format (1.2GB for images, 40MB for supplementary metadata, and 7MB for the malignancy ground truth). Primary access and challenge participation took place via **Kaggle**. The mandatory citation for the CC-BY-NC version is: International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. *International Skin Imaging Collaboration* [https://doi.org/10.34970/2024-slice-3d] (2024).

## URL

https://challenge2024.isic-archive.com/