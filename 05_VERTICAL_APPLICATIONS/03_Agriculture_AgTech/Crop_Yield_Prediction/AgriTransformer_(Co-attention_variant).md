# AgriTransformer (Co-attention variant)

## Description

AgriTransformer é uma arquitetura de *deep learning* baseada em *Transformer* que utiliza um mecanismo de **co-atenção (cross-modal attention)** para integrar dados multimodais (dados tabulares de manejo agrícola e índices de vegetação de imagens de satélite) para aprimorar a previsão de rendimento de culturas. O mecanismo de co-atenção permite que o modelo aprenda as interdependências dinâmicas entre as diferentes modalidades de dados, superando modelos que tratam as modalidades separadamente ou as concatenam de forma simples. O modelo foi desenvolvido para capturar as complexas interações não lineares entre as condições ambientais, práticas de manejo e respostas fisiológicas das plantas.

## Statistics

- **Melhor Desempenho:** Variantes com co-atenção.
- **Métricas:** MSE = 2.598 (Média), R² = 0.919 (Média).
- **Comparação:** Supera significativamente a Regressão Linear (R² = 0.704) e Redes Neurais Densas (R² = 0.884).
- **Dataset:** Telangana Crop Health Challenge (Índia).
- **Citações:** 3 (até a data de publicação do artigo em 2025).

## Features

- **Mecanismo de Co-Atenção (Cross-Modal Attention):** Permite a fusão eficiente de dados heterogêneos (tabulares e espectrais), aprendendo como as características de uma modalidade influenciam a outra.
- **Arquitetura Baseada em Transformer:** Utiliza a força dos *Transformers* para modelar dependências complexas e de longo alcance nos dados.
- **Entradas Multimodais:** Processa simultaneamente dados tabulares (tipo de cultura, irrigação, manejo) e índices de vegetação (NDVI, EVI, NDWI, GNDVI, SAVI, MSAVI) derivados de imagens de satélite Sentinel-2.
- **Modularidade:** Design modular que facilita a adaptabilidade e transferibilidade para diferentes tipos de culturas e regiões geográficas.

## Use Cases

- **Previsão de Rendimento de Culturas:** Estimação precisa da produção agrícola em nível de campo, crucial para a segurança alimentar e planejamento logístico.
- **Agricultura de Precisão:** Suporte à tomada de decisões em tempo real sobre irrigação, fertilização e manejo de culturas.
- **Modelagem de Interações Complexas:** Identificação e quantificação da influência de fatores de manejo (irrigação, tipo de cultura) sobre as condições de saúde da vegetação (VIs).
- **Monitoramento Agrícola:** Fornece uma base robusta para o monitoramento da saúde e desenvolvimento das culturas ao longo da estação de crescimento.

## Integration

O modelo foi implementado usando as *frameworks* **TensorFlow** e **Keras** (Python), com o **Google Earth Engine** sendo utilizado para a obtenção das imagens multiespectrais de satélite. A integração envolve:
1.  **Pré-processamento de Dados:** Normalização de dados tabulares e cálculo de Índices de Vegetação (VIs) a partir de imagens Sentinel-2.
2.  **Arquitetura:** Construção das camadas de *Embedding* para dados tabulares e VIs, seguidas pelo módulo de Co-Atenção.
3.  **Treinamento:** Otimizador Adam, *learning rate* inicial de 0.001, função de perda MSE.
*Não foi encontrado um repositório GitHub público com o código de implementação no momento da pesquisa, mas o artigo detalha a arquitetura para replicação.*

## URL

https://www.mdpi.com/2079-9292/14/12/2466