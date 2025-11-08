# Medical Imaging Datasets / Datasets de Imagem Médica

## 🇬🇧 English

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

## 🇧🇷 Português

### Visão Geral

Este diretório contém referências curadas aos maiores e mais importantes datasets de imagem médica para treinar e avaliar modelos de IA em saúde. Esses datasets são essenciais para desenvolver sistemas de IA médica robustos e generalizáveis.

---

## 📊 Datasets de Raio-X de Tórax em Grande Escala

### ⭐⭐⭐⭐ MIMIC-CXR - O Maior Dataset de Raio-X de Tórax

**Descrição:** MIMIC-CXR é o maior dataset de raio-X de tórax publicamente disponível, contendo mais de 377.000 imagens pareadas com relatórios radiológicos em texto livre.

**Características Principais:**
- **Tamanho:** 377.110 imagens de raio-X de tórax de 227.835 estudos de imagem
- **Pacientes:** 65.379 pacientes únicos
- **Relatórios:** Relatórios radiológicos em texto livre para todas as imagens
- **Instituição:** Instituto de Tecnologia de Massachusetts (MIT)
- **Período:** 2011-2016

**Tarefas:**
- Geração de relatórios radiológicos
- Classificação multi-rótulo de doenças (14 observações)
- Processamento de linguagem natural em relatórios médicos
- Aprendizado fracamente supervisionado a partir de relatórios

**Acesso:**
- **URL:** [https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr/)
- **Licença:** PhysioNet Credentialed Health Data License
- **Requisitos:** Completar treinamento CITI e assinar acordo de uso de dados

---

### ⭐⭐⭐⭐ CHEXPERT - Dataset de Competição de Raio-X de Tórax de Stanford

**Descrição:** CheXpert é um grande dataset de raios-X de tórax com rótulos de incerteza, projetado para classificação multi-rótulo de doenças torácicas.

**Características Principais:**
- **Tamanho:** 224.316 radiografias de tórax de 65.240 pacientes
- **Rótulos:** 14 observações (multi-rótulo com incerteza)
- **Tratamento de Incerteza:** Modelagem explícita de rótulos incertos, não mencionados e negativos
- **Instituição:** Universidade de Stanford
- **Competição:** Leaderboard ativo para benchmarking

**Tarefas:**
- Classificação multi-rótulo com incerteza
- Aprendizado fracamente supervisionado
- Tratamento de ruído em rótulos
- Classificação de vista frontal vs. lateral

**Acesso:**
- **URL:** [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **Licença:** Stanford University Dataset Research Use Agreement
- **Download:** Gratuito para uso em pesquisa

---

## 🧬 Datasets de Câncer e Patologia

### ⭐⭐⭐⭐⭐ TCGA - The Cancer Genome Atlas

**Descrição:** TCGA é um programa histórico de genômica do câncer que caracterizou molecularmente mais de 20.000 amostras de câncer primário e normais correspondentes, abrangendo 33 tipos de câncer.

**Características Principais:**
- **Tamanho:** 20.000+ imagens de patologia de lâmina inteira
- **Tipos de Câncer:** 33 tipos diferentes de câncer
- **Multi-Modal:** Dados de genômica, transcriptômica, proteômica e imagem
- **Instituição:** Institutos Nacionais de Saúde (NIH) / Instituto Nacional do Câncer (NCI)
- **Período:** 2006-2018

**Tarefas:**
- Diagnóstico e subtipagem de câncer
- Predição de prognóstico
- Correlação genômica-patológica
- Análise de sobrevivência
- Predição de mutação a partir de histopatologia

**Acesso:**
- **URL:** [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)
- **Portal:** [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
- **Licença:** Acesso aberto para pesquisa
- **Download:** Gratuito via GDC Data Portal

---

## 🔬 Datasets de Dermatologia e Lesões de Pele

### ⭐⭐⭐⭐ ISIC - International Skin Imaging Collaboration

**Descrição:** ISIC é a maior coleção pública de imagens dermoscópicas de lesões de pele com controle de qualidade, projetada para detecção de melanoma e análise de lesões de pele.

**Características Principais:**
- **Tamanho:** 33.000+ imagens dermoscópicas (e crescendo)
- **Tarefa:** Classificação de lesões de pele, detecção de melanoma
- **Instituição:** International Skin Imaging Collaboration
- **Desafios:** Desafios anuais ISIC em grandes conferências (MICCAI, CVPR)

**Tarefas:**
- Classificação binária (benigno vs. maligno)
- Classificação multi-classe de lesões
- Segmentação de lesões
- Detecção de atributos de lesões

**Acesso:**
- **URL:** [https://www.isic-archive.com/](https://www.isic-archive.com/)
- **Licença:** Creative Commons (CC BY-NC)
- **Download:** Gratuito para uso em pesquisa
- **API:** Disponível para acesso programático

---

## 📋 Comparação de Datasets

| Dataset | Tamanho | Modalidade | Tarefa | Instituição | Acesso |
|---|---|---|---|---|---|
| **MIMIC-CXR** | 377k+ imagens | Raio-X tórax | Geração relatórios, classificação | MIT | PhysioNet (credenciado) |
| **CHEXPERT** | 224k+ imagens | Raio-X tórax | Classificação multi-rótulo | Stanford | Gratuito (pesquisa) |
| **TCGA** | 20k+ lâminas | Patologia | Diagnóstico câncer, genômica | NIH/NCI | Gratuito (GDC Portal) |
| **ISIC** | 33k+ imagens | Dermoscopia | Detecção melanoma | ISIC | Gratuito (CC BY-NC) |

---

## Related Resources

- [Papers with Code - Medical Imaging Datasets](https://paperswithcode.com/datasets?task=medical-imaging)
- [Grand Challenge - Medical Imaging Challenges](https://grand-challenge.org/)
- [Kaggle - Medical Imaging Datasets](https://www.kaggle.com/datasets?search=medical+imaging)
