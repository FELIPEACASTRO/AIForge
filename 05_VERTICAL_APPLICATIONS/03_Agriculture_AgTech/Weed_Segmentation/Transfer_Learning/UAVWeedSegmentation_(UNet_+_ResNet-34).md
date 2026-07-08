# UAVWeedSegmentation (UNet + ResNet-34)

## Description

The resource is a source code repository that implements a Deep Learning solution for **Early Weed Segmentation** in UAV (Unmanned Aerial Vehicle) images of sorghum fields. The main focus is on handling images with motion blur, a common challenge in aerial captures. The main model used is a **UNet** architecture with a **ResNet-34** backbone, which is an example of applying Transfer Learning for semantic segmentation tasks in precision agriculture. The repository provides the code for training, retraining, prediction, and comparison of results with the Ground Truth. The associated publication is from 2022, but the repository is a key resource for the practical implementation of DL models in agriculture.

## Statistics

- **Best-Performing Model**: UNet + ResNet-34.
- **Per-Class Metrics (Test-Set)**:
    - **Background**: Precision (99.80%), Recall (99.93%), F1-Score (99.86%).
    - **Sorghum**: Precision (91.58%), Recall (86.10%), F1-Score (88.76%).
    - **Weed**: Precision (87.64%), Recall (72.71%), F1-Score (79.48%).
- **Macro Average**: Precision (93.01%), Recall (86.25%), F1-Score (89.37%).
- **Citation (2022 Paper)**: The paper "Deep Learning-based Early Weed Segmentation using Motion Blurred UAV Images of Sorghum Fields" was published in *Computers and Electronics in Agriculture* (DOI: 10.1016/j.compag.2022.107388). Although the publication is from 2022, the resource is a relevant and up-to-date DL implementation for the field.

## Features

- **Semantic Segmentation Model**: UNet with ResNet-34 as the feature extractor.
- **Focus on Motion-Blurred Images**: The model was trained and evaluated on a dataset that includes images with different degrees of motion blur.
- **Multiclass Segmentation**: Classifies pixels into three classes: Background, Sorghum, and Weed.
- **Implementation Scripts**: Includes scripts for patch generation, training (including retraining with the full set), prediction, and performance evaluation.
- **Transfer Learning**: Uses the ResNet-34 architecture, pre-trained on large datasets, as the basis for the segmentation task.

## Use Cases

- **Weed Mapping**: Creation of high-resolution segmentation maps to identify the exact location of weeds.
- **Variable-Rate Herbicide Application**: Use of segmentation maps to guide herbicide application only to infested areas, optimizing input use (Precision Agriculture).
- **Crop Monitoring**: Assessment of crop (sorghum) health and density at early stages.
- **Motion Blur Mitigation**: Demonstration of a robust model for handling the variable image quality of UAV captures.

## Integration

The repository provides a detailed installation and usage guide. Integration involves:
1.  **Cloning the Repository**: `git clone git@github.com:grimmlab/UAVWeedSegmentation.git`
2.  **Installing Dependencies**: `pip install -r requirements.txt` (Python 3.8, PyTorch 1.11.0, CUDA).
3.  **Downloading the Pre-trained Model**: Download the model from Mendeley Data and save it in `/models` with the name `model_unet_resnet34_dil0_bilin1_retrained.pt`.
4.  **Prediction on New Images**:
    ```bash
    python3 save_patches.py
    python3 predict_testset.py models/model_unet_resnet34_dil0_bilin1_retrained.pt test
    ```
5.  **Training New Models**: The `train.py` script allows training new architectures (fcn, unet, dlplus) with different *feature extractors* (resnet18, resnet34, resnet50, resnet101).

## URL

https://github.com/grimmlab/UAVWeedSegmentation