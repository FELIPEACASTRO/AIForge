# Snap-MAE (Snapshot Ensemble Learning into Masked Autoencoders)

## Description

Modelo de aprendizado auto-supervisionado (SSL) que integra o aprendizado de ensemble de *snapshot* (instantâneo) no processo de pré-treinamento do Masked Autoencoder (MAE). Utiliza um agendador de cosseno cíclico para capturar diversas representações do modelo em um único ciclo de treinamento, otimizando a eficiência computacional e melhorando o desempenho.

## Statistics

Supera consistentemente o MAE vanilla, ViT-S e ResNet-34 em todas as métricas de desempenho nos casos de uso testados. Reduz a carga computacional associada ao aprendizado de ensemble, produzindo múltiplos modelos pré-treinados a partir de uma única fase de pré-treinamento.

## Features

Integração de *snapshot ensemble learning* no MAE; Uso de agendador de cosseno cíclico; Otimização da eficiência computacional; Geração de múltiplos modelos pré-treinados diversos; Robustez no pré-treinamento com dados não rotulados.

## Use Cases

Classificação multi-rótulo de doenças torácicas pediátricas; Diagnóstico de doenças cardiovasculares.

## Integration

Implementação direta e eficaz para melhorar o pré-treinamento baseado em SSL em imagens médicas. (O artigo fornece a base teórica para implementação, mas o repositório oficial não foi encontrado na pesquisa).

## URL

https://www.nature.com/articles/s41598-025-15704-3