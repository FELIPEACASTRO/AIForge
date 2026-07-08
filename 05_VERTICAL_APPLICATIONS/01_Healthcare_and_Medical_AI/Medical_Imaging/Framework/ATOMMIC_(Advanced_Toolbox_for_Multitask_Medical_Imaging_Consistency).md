# ATOMMIC (Advanced Toolbox for Multitask Medical Imaging Consistency)

## Description

**ATOMMIC (Advanced Toolbox for Multitask Medical Imaging Consistency)** is an open-source, modular *framework* based on PyTorch Lightning, designed to facilitate the research and application of Artificial Intelligence methods in **Magnetic Resonance Imaging (MRI)**. It supports multiple tasks, including **accelerated MRI reconstruction (REC)**, **MRI segmentation (SEG)**, **quantitative MRI imaging (qMRI)**, and, crucially, **Multi-Task Learning (MTL)** to perform tasks simultaneously, such as joint reconstruction and segmentation. The framework aims to standardize workflows, ensure data interoperability, and enable the evaluation of *Deep Learning* models across diverse datasets and subsampling schemes. MTL is a core component, allowing the model to improve performance on joint tasks by learning shared representations. In addition, MTL has been successfully applied in multimodal models for the **simultaneous prediction of multiple chronic diseases** (such as diabetes, heart disease, stroke, and hypertension) from electronic health records, demonstrating its effectiveness in clinical risk assessment.

## Statistics

- **Year of Publication:** 2024 (Article: Computer Methods and Programs in Biomedicine, Volume 256, November 2024).
- **Citations:** 1 (on the original ScienceDirect article, as of Nov 2025).
- **Supported Models:** More than 25 state-of-the-art *Deep Learning* models.
- **Supported Datasets:** Native support for public datasets such as CC359, fastMRI, BraTS 2023, ISLES 2022, and SKM-TEA.
- **MTL Performance:** Demonstrates improved performance on joint tasks (e.g., reconstruction and segmentation) compared to single-task models.
- **Evaluation Metrics:** Uses standard field metrics such as SSIM, PSNR, NMSE (for reconstruction) and DICE, F1, IOU, HD95 (for segmentation).

## Features

- **Modularity and Extensibility:** Designed for the easy addition of new tasks, models, and datasets.
- **Support for Multiple MRI Tasks:** Includes collections for REC, SEG, qMRI, and MTL.
- **More than 25 SOTA Models:** Implements state-of-the-art models such as CIRIM, VarNet, UNet, and models specific to MTL (e.g., SERANet, MTLRS).
- **High-Performance Training:** Uses PyTorch Lightning for multi-GPU training, mixed precision, and hyperparameter optimization.
- **Data Interoperability:** Supports complex-valued and real-valued data, with extensive pre-processing transformations.
- **HuggingFace Integration:** Pre-trained models and *checkpoints* can be loaded and downloaded directly from HuggingFace.

## Use Cases

- **Accelerated MRI Reconstruction:** Reducing MRI acquisition time.
- **Medical Image Segmentation:** Segmentation of anatomical structures and lesions in MRI scans (e.g., brain tumors in BraTS).
- **Quantitative MRI Imaging:** Estimation of quantitative parameter maps.
- **MTL in Medical Imaging:** Simultaneous reconstruction and segmentation for consistency and improved performance.
- **Chronic Disease Prediction:** Application of MTL in multimodal networks (MAND) to simultaneously predict the risks of **diabetes mellitus, heart disease, stroke, and hypertension** from electronic health records.
- **Diagnosis and Prognosis in Oncology:** Use of MTL with CNNs and Vision Transformers to improve outcome prediction in patients with head and neck cancer.

## Integration

Installation is recommended via Conda and Pip:

```shell
conda create -n atommic python=3.10
conda activate atommic
pip install atommic
```

For research use, training and testing are performed through a `.yaml` configuration file (using Hydra and OmegaConf) and a simple command:

```shell
atommic run -c path-to-config-file.yaml
```

**Configuration Example (Conceptual for MTL):**
The `.yaml` file defines the complete *pipeline*, including the task (`MTL`), the model (e.g., `MTLRS`), the dataset (e.g., `SKM-TEA`), the subsampling parameters, transformations, optimizer, and scheduler.

**Docker Integration:**
A Docker image is available to facilitate deployment across different environments:

```shell
docker pull wdika/atommic
docker run --gpus all -it --rm -v /home/user/configs:/config atommic:latest atommic run -c /config/config.yaml
```

## URL

https://github.com/wdika/atommic
