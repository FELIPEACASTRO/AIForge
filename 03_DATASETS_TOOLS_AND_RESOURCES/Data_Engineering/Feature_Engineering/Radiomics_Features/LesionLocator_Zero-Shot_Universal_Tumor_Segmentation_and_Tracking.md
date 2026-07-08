# LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking

## Description

LesionLocator is an innovative deep learning framework for segmentation and longitudinal tracking of lesions (tumors) in whole-body 3D medical images. It is notable for its capability of **zero-shot universal tumor segmentation** (without specific training for new lesion types), using dense spatial *prompts* (such as points or boxes) to identify and track lesions across follow-up (4D) exams. The model was trained on an extensive dataset of 23,262 annotated medical scans, in addition to synthetic longitudinal data, which gives it high generalization. The framework establishes a new *benchmark* in universal segmentation and automated longitudinal lesion tracking.

## Statistics

Model trained on an extensive dataset of 23,262 annotated medical scans and synthetic longitudinal data. The training dataset spans 47 diverse datasets, with different imaging modalities and anatomical targets. The model achieves human-level performance in lesion segmentation.

## Features

Zero-Shot Universal Lesion Segmentation; 4D Longitudinal Lesion Tracking; Uses dense spatial *prompts* (point or box); Generalizable deep learning architecture; High segmentation performance (outperforms existing models by almost 10 Dice points); Open-source solution (model and 4D synthetic dataset).

## Use Cases

Automated segmentation and tracking of tumors in follow-up (longitudinal) exams; Support for diagnosis and monitoring of various tumoral lesions in whole-body 3D medical images; Research and development of new AI approaches in oncology and radiology.

## Integration

The code and the pre-trained model *checkpoint* are available on GitHub and Zenodo. The model is designed to be the first open-access one for universal lesion segmentation and automated longitudinal tracking. Usage involves applying the pre-trained model to new 3D medical images, providing spatial *prompts* (point or box) for the lesion of interest. The framework is compatible with lesion tracking across follow-up exams.

## URL

https://arxiv.org/abs/2502.20985