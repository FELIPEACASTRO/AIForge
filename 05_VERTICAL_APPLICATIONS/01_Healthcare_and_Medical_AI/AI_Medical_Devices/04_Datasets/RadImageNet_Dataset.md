# RadImageNet Dataset / Dataset RadImageNet

## 🇬🇧 English

### Overview

RadImageNet is a large-scale, open-access medical imaging database designed to significantly improve **transfer learning** performance on downstream medical imaging applications. It is one of the most important medical imaging datasets for pre-training deep learning models.

---

## 📊 Key Features

| Feature | Detail |
|---|---|
| **Size** | **1.35 million** annotated medical images |
| **Patients** | 131,872 unique patients |
| **Modalities** | Computed Tomography (CT), Magnetic Resonance Imaging (MRI), and Ultrasound (US) |
| **Anatomies** | 11 different anatomical regions (musculoskeletal, neurologic, oncologic, etc.) |
| **Pathologies** | 165 pathologic labels |
| **Purpose** | Pre-training Convolutional Neural Networks (CNNs) for transfer learning |
| **Performance** | RadImageNet pre-trained models consistently **outperform ImageNet** pre-trained models on various medical tasks. |

### Tasks Improved by RadImageNet

RadImageNet pre-trained models have shown superior performance in transfer learning for tasks such as:
- Thyroid nodule malignancy prediction on ultrasound
- Breast lesion classification on ultrasound
- ACL and meniscus tear detection on MR
- Pneumonia detection on chest radiographs
- SARS-CoV-2 detection and COVID-19 identification on chest CT
- Hemorrhage detection on head CT

### Pre-trained Models Available

The RadImageNet repository provides pre-trained models for popular architectures, including:
- ResNet50
- DenseNet121
- InceptionResNetV2
- InceptionV3

These models are trained **solely on medical images**, making them ideal starting points for new medical AI projects.

### Access

- **Official Website:** [https://www.radimagenet.com/](https://www.radimagenet.com/)
- **GitHub Repository:** [https://github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)
- **Data Access:** Available by request on the official website.

### Citation

```bibtex
@article{doi:10.1148/ryai.210315,
author = {Mei, Xueyan and others},
title = {RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning},
journal = {Radiology: Artificial Intelligence},
volume = {0},
number = {ja},
pages = {e210315},
year = {2022},
doi = {10.1148/ryai.210315}
}
```

---

## 🇧🇷 Português

### Visão Geral

RadImageNet é um grande banco de dados de imagens médicas de acesso aberto, projetado para melhorar significativamente o desempenho do **transfer learning** em aplicações downstream de imagens médicas. É um dos datasets mais importantes para o pré-treinamento de modelos de deep learning na área médica.

---

## 📊 Características Principais

| Característica | Detalhe |
|---|---|
| **Tamanho** | **1,35 milhão** de imagens médicas anotadas |
| **Pacientes** | 131.872 pacientes únicos |
| **Modalidades** | Tomografia Computadorizada (TC), Ressonância Magnética (RM) e Ultrassom (US) |
| **Anatomias** | 11 regiões anatômicas diferentes (musculoesquelético, neurológico, oncológico, etc.) |
| **Patologias** | 165 rótulos patológicos |
| **Propósito** | Pré-treinamento de Redes Neurais Convolucionais (CNNs) para transfer learning |
| **Desempenho** | Modelos pré-treinados no RadImageNet consistentemente **superam o ImageNet** em várias tarefas médicas. |

### Tarefas Melhoradas pelo RadImageNet

Modelos pré-treinados no RadImageNet demonstraram desempenho superior em transfer learning para tarefas como:
- Predição de malignidade de nódulo tireoidiano em ultrassom
- Classificação de lesão mamária em ultrassom
- Detecção de ruptura de LCA e menisco em RM
- Detecção de pneumonia em radiografias de tórax
- Detecção de SARS-CoV-2 e identificação de COVID-19 em TC de tórax
- Detecção de hemorragia em TC de cabeça

### Modelos Pré-treinados Disponíveis

O repositório RadImageNet fornece modelos pré-treinados para arquiteturas populares, incluindo:
- ResNet50
- DenseNet121
- InceptionResNetV2
- InceptionV3

Esses modelos são treinados **exclusivamente em imagens médicas**, tornando-os pontos de partida ideais para novos projetos de IA médica.

### Acesso

- **Site Oficial:** [https://www.radimagenet.com/](https://www.radimagenet.com/)
- **Repositório GitHub:** [https://github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)
- **Acesso aos Dados:** Disponível mediante solicitação no site oficial.

---

## Recursos Relacionados

- [HuggingFace - Lab-Rasool/RadImageNet](https://huggingface.co/Lab-Rasool/RadImageNet)
- [Pretrained RadImageNet Models](https://www.kaggle.com/datasets/ipythonx/notop-wg-radimagenet)
- [RadImageNet: An Open Radiologic Deep Learning Research Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)
