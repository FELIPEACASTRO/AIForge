# Medical Imaging Datasets

### Overview

This directory contains curated references to the largest and most important medical imaging datasets for training and evaluating AI models in healthcare. These datasets are essential for developing robust, generalizable medical AI systems.

---

## 📊 Large-Scale Chest X-Ray Datasets

### ⭐⭐⭐⭐ MIMIC-CXR - The Largest Chest X-Ray Dataset

**Description:** MIMIC-CXR is the largest publicly available chest X-ray dataset, containing over 377,000 images paired with free-text radiology reports.

**Key Features:**
- **Size:** 377,110 chest X-ray images from 227,835 imaging studies
- **Patients:** 65,379 unique patients
- **Reports:** Free-text radiology reports for all images
- **Institution:** Massachusetts Institute of Technology (MIT)
- **Time Period:** 2011-2016

**Tasks:**
- Radiology report generation
- Multi-label disease classification (14 observations)
- Natural language processing on medical reports
- Weakly-supervised learning from reports

**Access:**
- **URL:** [https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr/)
- **License:** PhysioNet Credentialed Health Data License
- **Requirements:** Complete CITI training and sign data use agreement

**Citation:**
```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and others},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={317},
  year={2019}
}
```

---

### ⭐⭐⭐⭐ CHEXPERT - Stanford Chest X-Ray Competition Dataset

**Description:** CheXpert is a large dataset of chest X-rays with uncertainty labels, designed for multi-label classification of thoracic diseases.

**Key Features:**
- **Size:** 224,316 chest radiographs from 65,240 patients
- **Labels:** 14 observations (multi-label with uncertainty)
- **Uncertainty Handling:** Explicit modeling of uncertain, unmentioned, and negative labels
- **Institution:** Stanford University
- **Competition:** Active leaderboard for benchmarking

**Tasks:**
- Multi-label classification with uncertainty
- Weakly-supervised learning
- Label noise handling
- Frontal vs. lateral view classification

**Labels (14 Observations):**
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

**Access:**
- **URL:** [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **License:** Stanford University Dataset Research Use Agreement
- **Download:** Free for research use

**Citation:**
```bibtex
@inproceedings{irvin2019chexpert,
  title={Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author={Irvin, Jeremy and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={590--597},
  year={2019}
}
```

---

## 🧬 Cancer and Pathology Datasets

### ⭐⭐⭐⭐⭐ TCGA - The Cancer Genome Atlas

**Description:** TCGA is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types.

**Key Features:**
- **Size:** 20,000+ whole-slide pathology images
- **Cancer Types:** 33 different cancer types
- **Multi-Modal:** Genomics, transcriptomics, proteomics, and imaging data
- **Institution:** National Institutes of Health (NIH) / National Cancer Institute (NCI)
- **Time Period:** 2006-2018

**Cancer Types Included:**
Breast, Lung, Prostate, Colorectal, Ovarian, Glioblastoma, Kidney, Liver, Pancreatic, Thyroid, Melanoma, and 22 others

**Tasks:**
- Cancer diagnosis and subtyping
- Prognosis prediction
- Genomic-pathologic correlation
- Survival analysis
- Mutation prediction from histopathology

**Access:**
- **URL:** [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)
- **Portal:** [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
- **License:** Open access for research
- **Download:** Free via GDC Data Portal

**Citation:**
```bibtex
@article{weinstein2013cancer,
  title={The cancer genome atlas pan-cancer analysis project},
  author={Weinstein, John N and others},
  journal={Nature Genetics},
  volume={45},
  number={10},
  pages={1113--1120},
  year={2013}
}
```

---

## 🔬 Dermatology and Skin Lesion Datasets

### ⭐⭐⭐⭐ ISIC - International Skin Imaging Collaboration

**Description:** ISIC is the largest public collection of quality-controlled dermoscopic images of skin lesions, designed for melanoma detection and skin lesion analysis.

**Key Features:**
- **Size:** 33,000+ dermoscopic images (and growing)
- **Task:** Skin lesion classification, melanoma detection
- **Institution:** International Skin Imaging Collaboration
- **Challenges:** Annual ISIC challenges at major conferences (MICCAI, CVPR)

**Lesion Types:**
- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosis
- Dermatofibroma
- Vascular lesion

**Tasks:**
- Binary classification (benign vs. malignant)
- Multi-class lesion classification
- Lesion segmentation
- Lesion attribute detection

**Access:**
- **URL:** [https://www.isic-archive.com/](https://www.isic-archive.com/)
- **License:** Creative Commons (CC BY-NC)
- **Download:** Free for research use
- **API:** Available for programmatic access

**Citation:**
```bibtex
@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic)},
  author={Codella, Noel and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}
```

---

## 📋 Dataset Comparison

| Dataset | Size | Modality | Task | Institution | Access |
|---|---|---|---|---|---|
| **MIMIC-CXR** | 377k+ images | Chest X-ray | Report generation, classification | MIT | PhysioNet (credentialed) |
| **CHEXPERT** | 224k+ images | Chest X-ray | Multi-label classification | Stanford | Free (research) |
| **TCGA** | 20k+ slides | Pathology | Cancer diagnosis, genomics | NIH/NCI | Free (GDC Portal) |
| **ISIC** | 33k+ images | Dermoscopy | Melanoma detection | ISIC | Free (CC BY-NC) |

---

## Related Resources

- [Papers with Code - Medical Imaging Datasets](https://paperswithcode.com/datasets?task=medical-imaging)
- [Grand Challenge - Medical Imaging Challenges](https://grand-challenge.org/)
- [Kaggle - Medical Imaging Datasets](https://www.kaggle.com/datasets?search=medical+imaging)
