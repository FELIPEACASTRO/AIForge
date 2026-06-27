# CO-ResNetRS50-SSL

## Description

Método de classificação de imagens semi-supervisionado (SSL) baseado em uma arquitetura ResNetRS50 aprimorada. O modelo integra o mecanismo de Atenção Coordenada (CA) para melhor extração de características posicionais e a Convolução Dinâmica Omni-Dimensional (ODConv) para aumentar a adaptabilidade dos kernels de convolução a diferentes alvos. O framework SSL é empregado para aumentar a capacidade de generalização e reduzir a dependência de grandes volumes de dados rotulados.

## Statistics

Acurácia de 90.38%, Precisão de 90.59%, e F1 Score de 90.19% (com 65.38 Milhões de parâmetros). Supera modelos de última geração (FasterNet-T1, Swin Transformer, etc.) com diferenças altamente significativas (p < 0.001).

## Features

Aprendizagem Semi-Supervisionada (SSL), Backbone ResNetRS50, Atenção Coordenada (CA), Convolução Dinâmica Omni-Dimensional (ODConv).

## Use Cases

Reconhecimento preciso do estágio de crescimento do arroz (arroz) em condições complexas de campo, otimizando a alocação de recursos (fertilização, irrigação) na agricultura de precisão.

## Integration

O modelo é baseado em ResNetRS50 e utiliza módulos CA e ODConv. A implementação provavelmente requer PyTorch ou TensorFlow, seguindo a arquitetura ResNetRS e a lógica de treinamento semi-supervisionado (por exemplo, pseudo-rotulagem). O artigo sugere o uso de imagens de 224x224 pixels.

## URL

https://www.sciencedirect.com/science/article/pii/S1161030125001273