# MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer

## Description

O **MMST-ViT** (Multi-Modal Spatial-Temporal Vision Transformer) é uma solução de aprendizado profundo desenvolvida para a **previsão de rendimento de culturas** (crop yield prediction) em nível de condado nos Estados Unidos. O modelo é notável por sua capacidade de integrar e processar dados multimodais e espaço-temporais, abordando os desafios impostos pelas variações meteorológicas de curto prazo e pelas mudanças climáticas de longo prazo no crescimento das culturas.

A arquitetura do MMST-ViT é composta por três componentes principais:
1.  **Multi-Modal Transformer:** Combina dados visuais de sensoriamento remoto (imagens Sentinel-2) com dados meteorológicos de curto prazo (WRF-HRRR) para modelar o impacto direto das condições climáticas sazonais.
2.  **Spatial Transformer:** Aprende a dependência espacial de alta resolução entre os condados, permitindo um rastreamento agrícola preciso.
3.  **Temporal Transformer:** Captura a dependência temporal de longo alcance, essencial para modelar o impacto das mudanças climáticas de longo prazo nas culturas.

O modelo também utiliza uma técnica de aprendizado contrastivo multimodal para o pré-treinamento, reduzindo a necessidade de supervisão humana extensiva.

## Statistics

O MMST-ViT demonstrou desempenho superior em comparação com seus equivalentes em experimentos extensivos em mais de 200 condados dos EUA.

**Métricas de Desempenho (Previsão de Soja):**
- **Root Mean Square Error (RMSE):** **3.9** (o mais baixo entre os modelos comparados)
- **R-squared ($R^2$):** **0.843** (o mais alto entre os modelos comparados)
- **Correlação (Corr):** **0.918**

**Citações:** O artigo foi publicado na **IEEE/CVF International Conference on Computer Vision (ICCV) em 2023** e possui um número significativo de citações (mais de 76, conforme o arXiv), indicando sua relevância na comunidade de pesquisa.

**Dataset:** Utiliza o **Tiny CropNet** e o **CropNet**, que abrangem dados de 2017 a 2022 em mais de 200 condados dos EUA.

## Features

- **Arquitetura Transformer Híbrida:** Combina Multi-Modal, Spatial e Temporal Transformers.
- **Multimodalidade:** Integra imagens de satélite (Sentinel-2), dados meteorológicos (WRF-HRRR) e dados de culturas (USDA Crop Dataset).
- **Conscientização Climática:** Projetado para capturar os impactos de variações climáticas de curto e longo prazo.
- **Pré-treinamento Contrastivo:** Utiliza aprendizado contrastivo multimodal para pré-treinamento eficiente.
- **Previsão em Nível de Condado:** Oferece previsões de alto nível de granularidade para grandes áreas geográficas.

## Use Cases

- **Previsão de Rendimento de Culturas:** Principal aplicação, focada em culturas como soja, milho e algodão em nível de condado.
- **Monitoramento Agrícola de Precisão:** O Spatial Transformer permite o rastreamento agrícola preciso e de alta resolução.
- **Análise de Impacto Climático:** Avaliação dos efeitos de variações meteorológicas de curto prazo e mudanças climáticas de longo prazo na produção agrícola.
- **Planejamento e Tomada de Decisão Agrícola:** Fornece informações valiosas para otimizar a produção e garantir o suprimento alimentar.

## Integration

A implementação oficial do MMST-ViT está disponível no GitHub, fornecendo o código-fonte em PyTorch e instruções detalhadas para pré-treinamento e ajuste fino (fine-tuning).

**Requisitos Principais:**
- `torch == 1.13.0`
- `torchvision == 0.14.0`
- `timm == 0.5.4`
- `numpy == 1.24.4`
- `pandas == 2.0.3`

**Exemplo de Uso (Ajuste Fino):**
O processo de ajuste fino para previsão de rendimento de culturas pode ser iniciado com o seguinte comando no ambiente PyTorch:

```python
# Instalar dependências
pip install -r requirements.txt

# Gerar arquivo de configuração JSON para o dataloader (exemplo para soja)
python config/build_config_soybean.py

# Executar o ajuste fino
python main_finetune_mmst_vit.py
```

O repositório também disponibiliza o dataset **Tiny CropNet** e o **CropNet** (uma extensão) no HuggingFace Datasets, facilitando a reprodução e o uso.

## URL

https://github.com/fudong03/MMST-ViT