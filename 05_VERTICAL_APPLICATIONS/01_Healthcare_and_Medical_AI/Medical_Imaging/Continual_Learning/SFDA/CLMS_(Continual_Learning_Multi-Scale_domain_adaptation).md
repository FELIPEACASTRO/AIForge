# CLMS (Continual Learning Multi-Scale domain adaptation)

## Description

**CLMS (Continual Learning Multi-Scale domain adaptation)** is a cutting-edge framework for adapting Deep Learning models to new clinical domains in medical imaging, even when the original training (source) data is not available (Source-Free Domain Adaptation - SFDA). CLMS integrates multi-scale reconstruction, continual learning (CL), and style alignment to overcome challenges such as error propagation, visual and structural feature misalignment, and catastrophic forgetting of prior knowledge. The goal is to ensure the safe and reliable deployment of AI models across diverse healthcare environments.

## Statistics

- **Prostate MRI Segmentation:** **10.87%** improvement in the Dice coefficient (compared to the state of the art).
- **Colonoscopy Polyp Segmentation:** **17.73%** improvement in the Dice coefficient.
- **Plus Disease Classification in Retinal Images:** **11.19%** improvement in AUC (Area Under the Curve).
- **Knowledge Preservation:** Preserved source knowledge for all tasks, avoiding catastrophic forgetting.
- **Citations:** Cited by 8 (as of December 2024, per the snippet).
- **Publication:** *Medical Image Analysis* (2025).

## Features

- **Source-Free Domain Adaptation (SFDA):** Adapts models to new unlabeled target domains without access to the original source data.
- **Continual Learning (CL):** Preserves source knowledge and avoids catastrophic forgetting when integrating new target-domain data.
- **Multi-Scale Reconstruction:** Addresses the visual and structural feature misalignment between domains.
- **Style Alignment:** Helps reduce data discrepancies between different clinical sites.
- **End-to-End Solution:** Complete framework for robust knowledge transfer and adaptation.

## Use Cases

- **Model Deployment in New Hospitals/Clinics:** Adapting AI models trained at one site for use at another, where the data characteristics (equipment, population) are different.
- **Robust Medical Image Segmentation:** Application in critical tasks such as segmentation of tumors, organs, or lesions across different imaging modalities (MRI, Colonoscopy, Retinography).
- **Adaptive Diagnosis:** Creation of diagnostic systems that can evolve and adapt to new patient populations or new imaging protocols without the need to retrain the model from scratch with the original data.

## Integration

CLMS is a research *framework*, and although the exact code was not extracted, the implementation is based on Deep Learning techniques and requires:
1.  A model pre-trained on the source domain (Source Domain).
2.  An unlabeled dataset from the new target domain (Target Domain).
3.  Implementation of the multi-scale reconstruction module, the continual learning strategy (to avoid forgetting), and the style alignment component.
The paper suggests that the implementation is done in Python, using standard Deep Learning libraries (such as PyTorch or TensorFlow), and it is a domain adaptation method. The source code is generally made available in GitHub repositories associated with the authors.

## URL

https://www.sciencedirect.com/science/article/pii/S1361841524003293