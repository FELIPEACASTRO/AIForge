# 🤖 AI Models for Biomass Estimation / Modelos de IA para Estimativa de Biomassa

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🤖 AI Models for Biomass Estimation

Production-ready AI models, foundation models, and research implementations for above-ground biomass (AGB) estimation using satellite imagery, drone data, and multi-modal inputs.

### Featured Models

---

## ⭐ 1. IBM Granite Geospatial Biomass

**Organization:** IBM Granite  
**URL:** https://huggingface.co/ibm-granite/granite-geospatial-biomass  
**License:** Apache-2.0 (Open Source)  
**Status:** Production-ready ✅

### Overview

The **granite-geospatial-biomass** model is a fine-tuned geospatial foundation model for predicting total above-ground biomass using optical satellite imagery. It is uniquely trained on data from **15 biomes across the globe**, making it one of the most comprehensive biomass estimation models available.

### Key Features

- **Architecture:** Swin-B transformer + UPerNet decoder
- **Pretraining:** SimMIM (self-supervised learning with masked reconstruction)
- **Training Data:** NASA HLS L30 (Harmonized Landsat-Sentinel 2) + GEDI L4A (Global Ecosystem Dynamics Investigation)
- **Coverage:** 15 biomes globally
- **Framework:** TerraTorch (open-source geospatial toolkit)
- **Downloads:** 404/month (active usage)
- **Community:** 46 likes, 3.44k followers

### Technical Details

**Backbone: Swin-B Transformer**
- Smaller starting patch size → higher effective resolution
- Windowed attention → better computational efficiency
- Hierarchical merging → useful inductive bias

**Decoder: UPerNet**
- Adapted for pixel-wise regression
- Fusion between transformer blocks (similar to Unet)
- Two Pixel Shuffle layers for upscaling

**Training Methodology:**
1. Acquire HLS data during leaf-on season
2. Analyze timeseries, select cloud-free pixels
3. Compute mean value per spectral band
4. Assemble composite image
5. Interpolate GEDI L4A biomass data to HLS grid
6. Align biomass points with HLS data

### How to Use

```python
from terratorch.cli_tools import LightningInferenceModel
from huggingface_hub import hf_hub_download

# Download model weights and config
ckpt_path = hf_hub_download(
    repo_id="ibm-granite/granite-geospatial-biomass", 
    filename="biomass_model.ckpt"
)
config_path = hf_hub_download(
    repo_id="ibm-granite/granite-geospatial-biomass", 
    filename="config.yaml"
)

# Load model
model = LightningInferenceModel.from_config(config_path, ckpt_path)

# Run inference
inference_results, input_file_names = model.inference_on_dir(<input_directory>)
```

### Experiments Available

1. **Zero-shot for all biomes** - No fine-tuning required
2. **Zero-shot for a single biome** - Biome-specific inference
3. **Few-shot for a single biome** - Fine-tune with limited data

### Resources

- **GitHub:** https://github.com/ibm-granite/granite-geospatial-biomass/
- **Getting Started Notebook:** Available on HuggingFace
- **Google Colab Notebook:** Available (high RAM required)
- **Papers:**
  - Fine-tuning of Geospatial Foundation Models (ArXiv 2406.19888)
  - TerraTorch: The Geospatial Foundation Models Toolkit (ArXiv 2503.20563)
  - Foundation Model (ArXiv 2310.18660)
- **IBM Blog:** https://research.ibm.com/blog/img-geospatial-studio-think

### Applications

- **Crop Yield Estimation** - Predict agricultural productivity
- **Forest Monitoring** - Monitor timber production
- **Carbon Sequestration** - Quantify carbon captured by nature
- **Climate Action** - Support nature-based climate solutions

### Citation

```bibtex
@misc{muszynski2024finetuninggeospatialfoundationmodels,
      title={Fine-tuning of Geospatial Foundation Models for Aboveground Biomass Estimation}, 
      author={Michal Muszynski and Levente Klein and Ademir Ferreira da Silva and Anjani Prasad Atluri and Carlos Gomes and Daniela Szwarcman and Gurkanwar Singh and Kewen Gu and Maciel Zortea and Naomi Simumba and Paolo Fraccaro and Shraddha Singh and Steve Meliksetian and Campbell Watson and Daiki Kimura and Harini Srinivasan},
      year={2024},
      url={https://arxiv.org/abs/2406.19888}, 
}
```

---

## 2. Vertify Biomass Model

**Organization:** Vertify.earth  
**URL:** https://huggingface.co/vertify/biomass-model  
**Focus:** Forest ecosystems  
**Data:** Multi-spectral satellite imagery

### Overview

This model predicts above-ground biomass (AGB) in **forest ecosystems** using multi-spectral satellite imagery. Developed by vertify.earth for the GIZ Forest project.

### Key Features

- **Focus:** Forest-specific biomass estimation
- **Data:** Multi-spectral satellite imagery
- **Partnership:** GIZ Forest project
- **Application:** Forest conservation and monitoring

---

## 3. MMCBE Dataset & Models

**URL:** https://huggingface.co/papers/2404.11256  
**Type:** Multi-modality dataset + benchmark models  
**Date:** April 2024

### Overview

**MMCBE (Multi-Modality Crop Biomass Estimation)** is a comprehensive dataset that includes:
- Drone images
- LiDAR point clouds
- Ground truth measurements

### Key Features

- **Multi-modal:** Combines visual and 3D spatial data
- **Benchmark:** State-of-the-art methods evaluated
- **Applications:** 3D crop modeling, biomass estimation
- **Research:** Enables advanced multi-modal learning

---

## Additional Models & Frameworks

### TerraTorch Framework

**URL:** https://github.com/ibm-granite/terratorch  
**Type:** Open-source geospatial AI toolkit  
**Organization:** IBM Granite

**Features:**
- Foundation models for geospatial data
- Easy model loading and inference
- Support for various geospatial tasks
- Integration with HuggingFace

### Deep Learning Architectures for Biomass

**Common Architectures:**
- **CNNs (Convolutional Neural Networks)** - Image-based estimation
- **Transformers (ViT, Swin)** - Global context understanding
- **UNet/UPerNet** - Pixel-wise regression
- **RNNs/LSTMs** - Temporal data integration

**Popular Frameworks:**
- **PyTorch** - Flexible deep learning
- **TensorFlow** - Production deployment
- **Keras** - Rapid prototyping
- **TerraTorch** - Geospatial-specific

---

<a name="português"></a>
## 🤖 Modelos de IA para Estimativa de Biomassa

Modelos de IA prontos para produção, foundation models e implementações de pesquisa para estimativa de biomassa acima do solo (AGB) usando imagens de satélite, dados de drones e entradas multi-modais.

### Modelos em Destaque

---

## ⭐ 1. IBM Granite Geospatial Biomass

**Organização:** IBM Granite  
**URL:** https://huggingface.co/ibm-granite/granite-geospatial-biomass  
**Licença:** Apache-2.0 (Código Aberto)  
**Status:** Pronto para produção ✅

### Visão Geral

O modelo **granite-geospatial-biomass** é um foundation model geoespacial ajustado para prever biomassa total acima do solo usando imagens de satélite ópticas. É unicamente treinado em dados de **15 biomas ao redor do globo**, tornando-o um dos modelos de estimativa de biomassa mais abrangentes disponíveis.

### Características Principais

- **Arquitetura:** Swin-B transformer + decodificador UPerNet
- **Pré-treinamento:** SimMIM (aprendizado auto-supervisionado com reconstrução mascarada)
- **Dados de Treinamento:** NASA HLS L30 (Harmonized Landsat-Sentinel 2) + GEDI L4A (Global Ecosystem Dynamics Investigation)
- **Cobertura:** 15 biomas globalmente
- **Framework:** TerraTorch (toolkit geoespacial de código aberto)
- **Downloads:** 404/mês (uso ativo)
- **Comunidade:** 46 likes, 3.44k seguidores

### Detalhes Técnicos

**Backbone: Swin-B Transformer**
- Tamanho de patch inicial menor → maior resolução efetiva
- Atenção em janelas → melhor eficiência computacional
- Fusão hierárquica → viés indutivo útil

**Decodificador: UPerNet**
- Adaptado para regressão pixel a pixel
- Fusão entre blocos transformer (similar ao Unet)
- Duas camadas Pixel Shuffle para upscaling

**Metodologia de Treinamento:**
1. Adquirir dados HLS durante estação de folhas
2. Analisar séries temporais, selecionar pixels sem nuvens
3. Calcular valor médio por banda espectral
4. Montar imagem composta
5. Interpolar dados de biomassa GEDI L4A para grade HLS
6. Alinhar pontos de biomassa com dados HLS

### Como Usar

```python
from terratorch.cli_tools import LightningInferenceModel
from huggingface_hub import hf_hub_download

# Baixar pesos do modelo e configuração
ckpt_path = hf_hub_download(
    repo_id="ibm-granite/granite-geospatial-biomass", 
    filename="biomass_model.ckpt"
)
config_path = hf_hub_download(
    repo_id="ibm-granite/granite-geospatial-biomass", 
    filename="config.yaml"
)

# Carregar modelo
model = LightningInferenceModel.from_config(config_path, ckpt_path)

# Executar inferência
inference_results, input_file_names = model.inference_on_dir(<input_directory>)
```

### Experimentos Disponíveis

1. **Zero-shot para todos os biomas** - Sem ajuste fino necessário
2. **Zero-shot para um único bioma** - Inferência específica por bioma
3. **Few-shot para um único bioma** - Ajuste fino com dados limitados

### Recursos

- **GitHub:** https://github.com/ibm-granite/granite-geospatial-biomass/
- **Notebook Getting Started:** Disponível no HuggingFace
- **Notebook Google Colab:** Disponível (RAM alta necessária)
- **Papers:**
  - Fine-tuning of Geospatial Foundation Models (ArXiv 2406.19888)
  - TerraTorch: The Geospatial Foundation Models Toolkit (ArXiv 2503.20563)
  - Foundation Model (ArXiv 2310.18660)
- **Blog IBM:** https://research.ibm.com/blog/img-geospatial-studio-think

### Aplicações

- **Estimativa de Produtividade Agrícola** - Prever produtividade agrícola
- **Monitoramento Florestal** - Monitorar produção de madeira
- **Sequestro de Carbono** - Quantificar carbono capturado pela natureza
- **Ação Climática** - Apoiar soluções climáticas baseadas na natureza

---

## 2. Vertify Biomass Model

**Organização:** Vertify.earth  
**URL:** https://huggingface.co/vertify/biomass-model  
**Foco:** Ecossistemas florestais  
**Dados:** Imagens de satélite multiespectrais

### Visão Geral

Este modelo prevê biomassa acima do solo (AGB) em **ecossistemas florestais** usando imagens de satélite multiespectrais. Desenvolvido por vertify.earth para o projeto GIZ Forest.

---

## 3. MMCBE Dataset & Models

**URL:** https://huggingface.co/papers/2404.11256  
**Tipo:** Dataset multi-modalidade + modelos benchmark  
**Data:** Abril 2024

### Visão Geral

**MMCBE (Multi-Modality Crop Biomass Estimation)** é um dataset abrangente que inclui:
- Imagens de drone
- Nuvens de pontos LiDAR
- Medições ground truth

---

**Última Atualização:** Novembro 2025  
**Total de Modelos:** 3+  
**Mantido por:** AIForge Community
