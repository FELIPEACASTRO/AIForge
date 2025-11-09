# 🤗 HuggingFace Medical Imaging AI Models

[English](#english) | [Português](#português)

---

## English

### Overview

HuggingFace hosts **160+ open-source medical imaging AI models** covering medical image segmentation, disease classification, medical image captioning, and EEG analysis. All models are freely available and can be integrated into research and clinical workflows.

### Key Statistics

- **160+ Models** (medical-imaging tag)
- **10+ Active Models** (updated in last 30 days)
- **Tasks:** Image Segmentation, Classification, Captioning, Object Detection
- **Applications:** Pneumonia, Brain Tumors, Thoracic Diseases, EEG Seizures

---

## TOP 10 HuggingFace Medical Imaging Models

### 1. **wanglab/MedSAM2** ⭐⭐⭐⭐⭐

**Task:** Image Segmentation  
**Updated:** July 2024  
**Description:** Medical image segmentation model based on SAM2 architecture  
**Use Case:** Segmentation of medical images across multiple modalities

**Key Features:**
- Based on Meta's Segment Anything Model 2 (SAM2)
- Supports multiple medical imaging modalities
- High accuracy for organ and lesion segmentation

**HuggingFace:** https://huggingface.co/wanglab/MedSAM2

---

### 2. **WafaaFraih/medical-image-captioning-roco** ⭐⭐⭐⭐⭐

**Task:** Image Captioning  
**Parameters:** 0.2B  
**Dataset:** ROCO (Radiology Objects in Context)  
**Use Case:** Automatic generation of medical image captions

**Key Features:**
- Trained on ROCO dataset (81,000+ radiology images)
- Generates clinically relevant captions
- Supports radiological image interpretation

**HuggingFace:** https://huggingface.co/WafaaFraih/medical-image-captioning-roco

---

### 3. **dlopez350/pneumonia_detector** ⭐⭐⭐⭐⭐

**Task:** Image Classification  
**Updated:** 1 day ago (ACTIVE)  
**Use Case:** Pneumonia detection from chest X-rays

**Key Features:**
- Recently updated (highly maintained)
- Binary classification (pneumonia vs normal)
- Optimized for chest X-ray images

**HuggingFace:** https://huggingface.co/dlopez350/pneumonia_detector

---

### 4. **microsoft/mri-autoencoder-v0.1** ⭐⭐⭐⭐

**Task:** MRI Image Encoding/Reconstruction  
**Updated:** April 8, 2024  
**Company:** Microsoft  
**Use Case:** MRI image compression and reconstruction

**Key Features:**
- Developed by Microsoft Research
- Autoencoder architecture for MRI
- Enables efficient MRI image storage and transmission

**HuggingFace:** https://huggingface.co/microsoft/mri-autoencoder-v0.1

---

### 5. **ziaddBou/pneumodoc-model** ⭐⭐⭐⭐

**Task:** Pneumonia Documentation and Classification  
**Updated:** April 18, 2024  
**Use Case:** Pneumonia detection and documentation

**Key Features:**
- Combines detection and documentation
- Clinical workflow integration
- Multi-class pneumonia classification

**HuggingFace:** https://huggingface.co/ziaddBou/pneumodoc-model

---

### 6. **trustworthy-ai/Federated-Learning-Disentanglement** ⭐⭐⭐⭐

**Task:** Image Segmentation  
**Updated:** July 13, 2024  
**Focus:** Federated Learning for Medical Imaging  
**Use Case:** Privacy-preserving medical image segmentation

**Key Features:**
- Federated learning approach (privacy-preserving)
- Disentanglement techniques
- Suitable for multi-institutional collaborations

**HuggingFace:** https://huggingface.co/trustworthy-ai/Federated-Learning-Disentanglement

---

### 7. **ummtushar/thoracic-disease-classifier** ⭐⭐⭐⭐

**Task:** Image Classification  
**Updated:** September 7, 2024  
**Use Case:** Thoracic disease detection from chest X-rays

**Key Features:**
- Multi-class thoracic disease classification
- Trained on ChestX-ray14 dataset
- Detects 14+ thoracic pathologies

**HuggingFace:** https://huggingface.co/ummtushar/thoracic-disease-classifier

---

### 8. **izeeek/resnet18_pneumonia_classifier** ⭐⭐⭐⭐

**Task:** Image Classification  
**Architecture:** ResNet18  
**Updated:** September 13, 2024  
**Use Case:** Pneumonia classification from chest X-rays

**Key Features:**
- ResNet18 architecture (lightweight)
- Fast inference
- Suitable for edge deployment

**HuggingFace:** https://huggingface.co/izeeek/resnet18_pneumonia_classifier

---

### 9. **ThomasCdnns/EEG-Seizure-Detection** ⭐⭐⭐⭐

**Task:** Image Classification (EEG signals as images)  
**Updated:** December 9, 2024  
**Use Case:** Seizure detection from EEG signals

**Key Features:**
- EEG signal classification
- Seizure vs non-seizure detection
- Supports epilepsy diagnosis

**HuggingFace:** https://huggingface.co/ThomasCdnns/EEG-Seizure-Detection

---

### 10. **pavankm96/brain_tumor_det** ⭐⭐⭐⭐

**Task:** Image Classification  
**Updated:** November 8, 2024  
**Use Case:** Brain tumor detection from MRI

**Key Features:**
- Binary classification (tumor vs no tumor)
- MRI-based detection
- Supports brain cancer screening

**HuggingFace:** https://huggingface.co/pavankm96/brain_tumor_det

---

## Models by Application

### Pneumonia Detection (3+ models)
- dlopez350/pneumonia_detector
- ziaddBou/pneumodoc-model
- izeeek/resnet18_pneumonia_classifier

### Brain Imaging
- pavankm96/brain_tumor_det
- icobrain (commercial, not on HuggingFace)

### Thoracic Diseases
- ummtushar/thoracic-disease-classifier

### EEG Analysis
- ThomasCdnns/EEG-Seizure-Detection

### Medical Image Segmentation
- wanglab/MedSAM2
- trustworthy-ai/Federated-Learning-Disentanglement

### Medical Image Captioning
- WafaaFraih/medical-image-captioning-roco

### MRI Analysis
- microsoft/mri-autoencoder-v0.1

---

## Resources

- **HuggingFace Medical Imaging:** https://huggingface.co/models?other=medical-imaging
- **Total Models:** 160+
- **Last Checked:** November 2025

---

## Português

### Visão Geral

O HuggingFace hospeda **160+ modelos de IA de imagem médica open-source** cobrindo segmentação de imagens médicas, classificação de doenças, legendagem de imagens médicas e análise de EEG. Todos os modelos são gratuitos e podem ser integrados em fluxos de trabalho de pesquisa e clínicos.

### Estatísticas Principais

- **160+ Modelos** (tag medical-imaging)
- **10+ Modelos Ativos** (atualizados nos últimos 30 dias)
- **Tarefas:** Segmentação de Imagem, Classificação, Legendagem, Detecção de Objetos
- **Aplicações:** Pneumonia, Tumores Cerebrais, Doenças Torácicas, Convulsões EEG

---

**Last Updated:** November 2025  
**Source:** HuggingFace  
**Total Models:** 160+
