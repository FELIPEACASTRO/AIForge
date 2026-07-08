# RadImageNet (Pre-trained Models)

## Description

**RadImageNet** is a set of convolutional neural network (CNN) models pre-trained exclusively on a large dataset of medical (radiological) images. The goal is to provide a more effective starting point for Transfer Learning (TL) in Medical Imaging applications, overcoming the limitations of models pre-trained on ImageNet (natural images). The original RadImageNet dataset contains **1.35 million annotated images** from Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and Ultrasound, covering 3 modalities, 11 anatomies, and 165 pathologies. Recent studies (2023-2025) indicate that RadImageNet models generally demonstrate superior or comparable performance to ImageNet on radiological tasks, especially in scenarios with limited data, and offer better interpretability.

## Statistics

**Dataset Size:** 1.35 million medical images (CT, MRI, Ultrasound). **Pre-trained Models:** ResNet50, DenseNet121, InceptionResNetV2, InceptionV3. **Performance (Top1/Top5 Accuracy on the RadImageNet dataset):** InceptionResNetV2: 74.0% / 94.3%; ResNet50: 72.3% / 94.1%; DenseNet121: 73.1% / 96.1%; InceptionV3: 73.2% / 92.7%. **Citations:** The main paper (DOI: 10.1148/ryai.210315) has been cited more than 388 times (data from 2022). **Comparison:** RadImageNet models outperform or match the performance of ImageNet models on radiological tasks, with advantages in limited-data scenarios.

## Features

**Domain-Specific Pre-training:** Trained exclusively on radiological images, which allows the models to learn features specific to the medical domain. **Modality Diversity:** Includes CT, MRI, and Ultrasound images. **Popular Architectures:** Pre-trained models available for widely used architectures such as ResNet50, DenseNet121, InceptionResNetV2, and InceptionV3. **Better Interpretability:** Demonstrates better interpretability compared to ImageNet models.

## Use Cases

**Lesion Classification:** Classification of breast lesions on ultrasound. **Pathology Detection:** Detection of anterior cruciate ligament (ACL) and meniscus tears on MRI. **Pulmonary Diagnosis:** Detection of pneumonia on chest radiographs and identification of SARS-CoV-2 on chest CT. **Hemorrhage Detection:** Detection of hemorrhage on head CT. **Malignancy Prediction:** Prediction of malignancy of thyroid nodules on ultrasound.

## Integration

**Model Availability:** The pre-trained models for TensorFlow and PyTorch are available via Google Drive (links in the official GitHub repository). **Code Example (PyTorch):** The official GitHub repository (BMEII-AI/RadImageNet) contains an example notebook (`pytorch_example.ipynb`) and training scripts (`*_train.py`) for various medical applications, demonstrating the fine-tuning process. **Usage (Conceptual):** Load the pre-trained RadImageNet model and perform fine-tuning on a dataset specific to the desired medical task, adjusting the learning rate and the number of frozen layers.

## URL

https://github.com/BMEII-AI/RadImageNet