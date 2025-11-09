# Múltiplos Recursos (Agri-GNN, HV-GNN, OpenAg, UAV/Random Forest)

## Description

Pesquisa abrangente sobre a aplicação de Redes Neurais Gráficas (GNNs) em Sistemas Agrícolas, com foco em artigos e frameworks recentes (2023-2025). Foram identificados modelos específicos para previsão de produtividade e detecção de pragas, além de um framework de Inteligência Artificial Geral Agrícola (AGI) que utiliza GNNs para grafos de conhecimento neurais. Um recurso adicional de estimativa de biomassa usando Machine Learning (Random Forest) e sensoriamento remoto por UAV também foi incluído devido à relevância para o tema de Biomassa e Agropecuária.

## Statistics

Agri-GNN: R² = 0.876 (Previsão de Produtividade em Iowa). HV-GNN: Precisão de 93.6625% (Detecção de Pragas em Café). UAV/Random Forest: R² = 0.76 (Estimativa de Biomassa Forrageira em Nebraska). OpenAg: Artigo de 2025 (v2 em Julho/2025).

## Features

GNNs Genotípico-Topológicas (Agri-GNN); GNNs de Visão Híbrida (HV-GNN); Grafos de Conhecimento Neurais (OpenAg); Previsão de Produtividade (R²=0.876); Detecção de Pragas (Precisão 93.66%); Estimativa de Biomassa (R²=0.76).

## Use Cases

Previsão otimizada da produtividade da colheita; Controle proativo de pragas e doenças em plantações; Suporte à decisão agrícola escalável e localmente relevante; Mapeamento espacial e temporal da biomassa forrageira.

## Integration

Os modelos Agri-GNN e HV-GNN são baseados em arquiteturas GNN (GraphSAGE e Visão Híbrida, respectivamente) e requerem a construção de grafos a partir de dados agrícolas (parcelas, genótipos, imagens). O framework OpenAg utiliza GNNs para o seu 'neural agricultural knowledge graph'. A integração geralmente envolve bibliotecas de GNN como PyTorch Geometric.

## URL

https://arxiv.org/abs/2310.13037, https://www.nature.com/articles/s41598-025-96523-4, https://arxiv.org/abs/2506.04571, https://www.sciencedirect.com/science/article/pii/S157495412500370X