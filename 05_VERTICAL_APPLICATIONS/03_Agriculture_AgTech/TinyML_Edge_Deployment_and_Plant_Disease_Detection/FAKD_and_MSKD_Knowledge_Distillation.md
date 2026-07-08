# FLOPs-Aware Knowledge Distillation (FAKD) framework and Multistage Knowledge Distillation (MSKD) method

## Description

Comprehensive research on the application of Knowledge Distillation for deploying AI models on edge devices (Edge AI) in agriculture. The identified resources focus on optimizing Deep Learning models for plant disease detection and other TinyML applications in resource-constrained environments.

## Statistics

FAKD: Accuracy on the PlantVillage dataset improved from 92.77% to 96.55% post-FAKD, with a tradeoff down to 90.15% after optimization for deployment on 1MB PSRAM. MSKD: The YOLOR-Light-v2 (distilled) model achieved 60.4% mAP@.5 on the PlantDoc dataset, with 20.5M parameters and 20.3GFLOPs, outperforming the Teacher model (YOLOR) in efficiency.

## Features

Two main resources were identified: 1) The FAKD framework, which uses FLOPs as a regularization term to optimize the computational efficiency of TinyML models (MobileNetV2) for constrained hardware (ESP32). 2) The MSKD method, which employs multistage knowledge distillation (backbone, neck, head) to improve the accuracy of lightweight object detection models (YOLOR-Light-v2) for plant disease diagnosis.

## Use Cases

Deployment of efficient and accurate models on low-power devices in resource-constrained environments (smart agriculture), such as miniaturized field equipment, IoT sensors, and Unmanned Aerial Vehicles (UAVs) for plant disease diagnosis.

## Integration

The MSKD method has implementation code available on GitHub (https://github.com/QDH/MSKD) and is based on PyTorch. The FAKD framework mentions optimization for deployment on hardware such as ESP32, indicating a focus on TinyML development tools.

## URL

https://ieeexplore.ieee.org/document/10933276/ and https://spj.science.org/doi/10.34133/plantphenomics.0062
