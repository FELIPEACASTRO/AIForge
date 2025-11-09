# Técnicas de Normalização em Redes Neurais (BatchNorm, LayerNorm, GroupNorm e Variantes)

## Description

A normalização é uma técnica fundamental em Deep Learning para estabilizar e acelerar o treinamento de redes neurais profundas, reduzindo o problema de *Internal Covariate Shift*. As principais técnicas (Batch, Layer, Instance e Group Norm) diferem na forma como calculam a média e a variância para normalizar as ativações, impactando sua eficácia em diferentes arquiteturas e tamanhos de lote.

## Statistics

BatchNorm é o padrão para CNNs com grandes lotes; LayerNorm é o padrão para RNNs e Transformers (NLP) devido à independência do tamanho do lote; GroupNorm supera BatchNorm em tarefas de visão computacional com lotes pequenos (e.g., detecção de objetos e segmentação semântica); InstanceNorm é altamente eficaz em tarefas de transferência de estilo de imagem.

## Features

BatchNorm: Normaliza ao longo da dimensão do lote. LayerNorm: Normaliza ao longo das dimensões do recurso (C, H, W) para cada amostra. InstanceNorm: Normaliza ao longo das dimensões espaciais (H, W) para cada canal e amostra. GroupNorm: Divide os canais em grupos e normaliza dentro de cada grupo para cada amostra. Variantes: RMSNorm (simplificação do LayerNorm, usado em modelos como o Llama) e Batch Layer Normalization (BLN, combinação de BN e LN).

## Use Cases

BatchNorm: Classificação de imagens (CNNs como ResNet, VGG) com lotes grandes. LayerNorm: Modelos de linguagem (Transformers como BERT, GPT), Redes Neurais Recorrentes (RNNs). InstanceNorm: Transferência de estilo de imagem, Geração de Imagens (GANs). GroupNorm: Tarefas de Visão Computacional com restrições de memória ou lotes pequenos (e.g., Fine-tuning em modelos pré-treinados).

## Integration

A integração é tipicamente feita como uma camada na arquitetura da rede neural. Exemplos em PyTorch:\n\n```python\nimport torch.nn as nn\n\n# Batch Normalization (para 2D, e.g., CNNs)\nbn = nn.BatchNorm2d(num_features=64)\n\n# Layer Normalization (para a última dimensão, e.g., em Transformers)\nln = nn.LayerNorm(normalized_shape=768)\n\n# Group Normalization (com 32 grupos)\ngn = nn.GroupNorm(num_groups=32, num_channels=64)\n\n# Instance Normalization (para 2D)\nin = nn.InstanceNorm2d(num_features=64)\n```\n\n**Nota:** A escolha da camada de normalização depende da arquitetura e do problema. Por exemplo, em Transformers, `nn.LayerNorm` é preferido por ser independente do lote.

## URL

https://arxiv.org/abs/1803.08494 (Group Normalization Paper) | https://docs.pytorch.org/docs/stable/nn.html#normalization-layers (PyTorch Docs)