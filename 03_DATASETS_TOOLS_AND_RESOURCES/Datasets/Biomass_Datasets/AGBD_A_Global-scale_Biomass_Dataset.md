# AGBD: A Global-scale Biomass Dataset

## Description

O **AGBD (A Global-scale Biomass Dataset)** é um novo conjunto de dados de referência global de biomassa acima do solo (AGB) projetado para treinamento de modelos de aprendizado de máquina. Ele combina dados de referência de AGB de campo com dados de sensoriamento remoto multimodais de séries temporais do satélite GEDI, Sentinel-1 e PALSAR-2. O dataset é representativo de diversas tipologias de vegetação e abrange vários anos, o que o torna ideal para modelos que exploram a dimensão temporal. O objetivo é fornecer um benchmark de alta resolução (10m) e globalmente representativo para a estimativa de AGB, superando as limitações de datasets existentes que são muito localizados ou de baixa resolução. O dataset é acompanhado por modelos de benchmark e está publicamente disponível.

## Statistics

- **Cobertura:** Global, abrangendo diversas tipologias de vegetação.
- **Resolução:** Mapa de predições de AGB de 10 metros.
- **Dados de Entrada:** Combinação de dados de AGB do GEDI, séries temporais de Sentinel-1 e PALSAR-2, mapa de densidade de dossel, mapa de elevação e mapa de classificação de uso do solo.
- **Tamanho Estimado:** O dataset completo é grande, com o artigo mencionando que o conjunto de dados de referência de AGB (AGB reference data) é cerca de 60 vezes maior que o ImageNet, o que exigiria aproximadamente 70TB de armazenamento. No entanto, a versão disponível para ML é uma versão pré-processada e mais acessível.
- **Período:** Abrange vários anos (séries temporais).

## Features

- **Global e Representativo:** Cobre diversas tipologias de vegetação em escala global.
- **Multimodal e Séries Temporais:** Combina dados de AGB do GEDI com séries temporais de Sentinel-1 (radar) e PALSAR-2 (radar), além de mapas de densidade de dossel, elevação e classificação de uso do solo.
- **Alta Resolução:** Oferece um mapa de predições de AGB de alta resolução (10m) para toda a área de cobertura do dataset.
- **Pronto para Machine Learning:** O dataset é pré-processado e compatível com frameworks como TensorFlow e PyTorch.
- **Benchmark:** Inclui modelos de benchmark rigorosamente testados.

## Use Cases

- **Treinamento de Modelos de AGB:** Utilizado para treinar modelos de aprendizado de máquina (ML) para estimativa de Biomassa Acima do Solo (AGB) em escala global.
- **Benchmark:** Serve como linha de base para validar a precisão e confiabilidade de novos modelos de estimativa de AGB.
- **Melhoria da Performance Regional:** Permite que pesquisadores ajustem (fine-tune) modelos globais com dados de referência locais para melhorar a precisão e performance regional.
- **Estudos de Carbono e Biodiversidade:** Suporta a avaliação de estoques de carbono florestal e a estrutura da biodiversidade.

## Integration

O dataset está hospedado no HuggingFace (Lhoest et al., 2021) e pode ser baixado e usado com as seguintes linhas de código (Python):

```python
# Instalação da biblioteca datasets
!pip install datasets

# Carregamento do dataset AGBD
from datasets import load_dataset
dataset = load_dataset("pre-eth/AGBD", streaming=True)["train"] # ou test, val
```

## URL

https://isprs-annals.copernicus.org/articles/X-G-2025/829/2025/isprs-annals-X-G-2025-829-2025.pdf