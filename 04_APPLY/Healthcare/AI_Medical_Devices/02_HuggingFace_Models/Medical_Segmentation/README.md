# Medical Segmentation Models / Modelos de Segmentação Médica

## 🇬🇧 English

### Overview

This directory contains state-of-the-art foundation models for universal medical image segmentation across multiple modalities and anatomical structures.

---

## ⭐ MedSAM - Segment Anything in Medical Images

### Description
MedSAM is a revolutionary foundation model designed for universal medical image segmentation, enabling accurate segmentation across diverse imaging modalities and anatomical structures without task-specific fine-tuning.

### Key Features
- **Universal Segmentation:** Works across 10 imaging modalities without retraining
- **Large-Scale Dataset:** Trained on 1,570,263 medical image-mask pairs
- **Multi-Modal Support:** CT, MRI, Endoscopy, Ultrasound, Pathology, Fundus, Dermoscopy, Mammography, OCT, X-ray
- **Cancer Coverage:** Over 30 cancer types
- **Superior Performance:** Outperforms SAM, U-Net, and DeepLabV3+ on medical imaging tasks

### Performance
- **Validation:** 86 internal + 60 external validation tasks
- **Metric:** Median Dice Similarity Coefficient (DSC)
- **Strengths:** Better boundary segmentation, especially on challenging targets with weak boundaries or low contrast

### Publication
- **Journal:** Nature Communications (2024)
- **Article Number:** 654
- **Citations:** 1,907+
- **DOI:** [10.1038/s41467-024-44824-z](https://www.nature.com/articles/s41467-024-44824-z)

### Authors
Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, Bo Wang

### Resources
- **GitHub:** [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
  - ⭐ 3,900+ stars
  - 🍴 528 forks
  - 👥 9 contributors
- **License:** Apache-2.0
- **Latest Release:** v1.0.0 (January 2, 2024)

### Available Tools
- Model checkpoint
- CoLab tutorial
- Jupyter notebook
- GUI (PyQt5)
- Training scripts (multi-GPU support)
- Pre-processing scripts
- Demo dataset (FLARE22Train)

---

## ⭐⭐ MedSAM2 - 3D and Video Segmentation

### Description
MedSAM2 is the next generation of MedSAM, extending capabilities to 3D medical imaging and video segmentation tasks.

### Key Features
- **3D Segmentation:** Native support for volumetric medical imaging
- **Video Segmentation:** Temporal consistency for medical video analysis
- **10x Faster:** LiteMedSAM variant runs 10x faster than original MedSAM
- **3D Slicer Plugin:** Seamless integration with medical imaging software

### Release Information
- **Release Date:** April 7, 2025
- **GitHub:** [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) (same repository)

### CVPR 2025 Challenges
- **Interactive 3D Biomedical Image Segmentation**
- **Text-guided 3D Biomedical Image Segmentation**

### CVPR 2024 Challenge
- **MedSAM on Laptop:** Optimized for resource-constrained environments

---

## 🇧🇷 Português

### Visão Geral

Este diretório contém modelos de fundação estado-da-arte para segmentação universal de imagens médicas em múltiplas modalidades e estruturas anatômicas.

---

## ⭐ MedSAM - Segmente Qualquer Coisa em Imagens Médicas

### Descrição
MedSAM é um modelo de fundação revolucionário projetado para segmentação universal de imagens médicas, permitindo segmentação precisa em diversas modalidades de imagem e estruturas anatômicas sem ajuste fino específico para cada tarefa.

### Características Principais
- **Segmentação Universal:** Funciona em 10 modalidades de imagem sem retreinamento
- **Dataset em Grande Escala:** Treinado em 1.570.263 pares imagem-máscara médicos
- **Suporte Multi-Modal:** CT, RM, Endoscopia, Ultrassom, Patologia, Fundo de olho, Dermoscopia, Mamografia, OCT, Raio-X
- **Cobertura de Câncer:** Mais de 30 tipos de câncer
- **Desempenho Superior:** Supera SAM, U-Net e DeepLabV3+ em tarefas de imagem médica

### Desempenho
- **Validação:** 86 tarefas internas + 60 tarefas externas de validação
- **Métrica:** Coeficiente de Similaridade de Dice (DSC) mediano
- **Pontos Fortes:** Melhor segmentação de bordas, especialmente em alvos desafiadores com bordas fracas ou baixo contraste

### Publicação
- **Revista:** Nature Communications (2024)
- **Número do Artigo:** 654
- **Citações:** 1.907+
- **DOI:** [10.1038/s41467-024-44824-z](https://www.nature.com/articles/s41467-024-44824-z)

### Autores
Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, Bo Wang

### Recursos
- **GitHub:** [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
  - ⭐ 3.900+ estrelas
  - 🍴 528 forks
  - 👥 9 contribuidores
- **Licença:** Apache-2.0
- **Última Versão:** v1.0.0 (2 de janeiro de 2024)

### Ferramentas Disponíveis
- Checkpoint do modelo
- Tutorial CoLab
- Jupyter notebook
- GUI (PyQt5)
- Scripts de treinamento (suporte multi-GPU)
- Scripts de pré-processamento
- Dataset de demonstração (FLARE22Train)

---

## ⭐⭐ MedSAM2 - Segmentação 3D e de Vídeo

### Descrição
MedSAM2 é a próxima geração do MedSAM, estendendo as capacidades para tarefas de imagem médica 3D e segmentação de vídeo.

### Características Principais
- **Segmentação 3D:** Suporte nativo para imagens médicas volumétricas
- **Segmentação de Vídeo:** Consistência temporal para análise de vídeo médico
- **10x Mais Rápido:** Variante LiteMedSAM executa 10x mais rápido que o MedSAM original
- **Plugin 3D Slicer:** Integração perfeita com software de imagem médica

### Informações de Lançamento
- **Data de Lançamento:** 7 de abril de 2025
- **GitHub:** [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) (mesmo repositório)

### Desafios CVPR 2025
- **Segmentação Interativa de Imagem Biomédica 3D**
- **Segmentação de Imagem Biomédica 3D Guiada por Texto**

### Desafio CVPR 2024
- **MedSAM on Laptop:** Otimizado para ambientes com recursos limitados

---

## Citation

```bibtex
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}
```

---

## Related Resources

- [Nature Communications Paper](https://www.nature.com/articles/s41467-024-44824-z)
- [GitHub Repository](https://github.com/bowang-lab/MedSAM)
- [CoLab Tutorial](https://colab.research.google.com/drive/1hhNdQAVMPBbPQlJsrpfMBwJvTqbJzjcX)
- [CVPR 2025 Challenges](https://www.synapse.org/#!Synapse:syn53708126/wiki/)
