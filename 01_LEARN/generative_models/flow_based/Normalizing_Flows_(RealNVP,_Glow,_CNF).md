# Normalizing Flows (RealNVP, Glow, CNF)

## Description

**Normalizing Flows (NFs)** são uma classe de modelos generativos que constroem distribuições de probabilidade complexas a partir de uma distribuição base simples (geralmente Gaussiana) através de uma sequência de transformações invertíveis e diferenciáveis [1] [2]. Sua principal proposta de valor é permitir a **avaliação exata e eficiente da densidade de probabilidade (likelihood)** e a **amostragem/inferência exatas** de variáveis latentes, algo que modelos como VAEs e GANs não oferecem [3]. O espaço latente é bem-comportado e tem a mesma dimensionalidade do espaço de dados. O **RealNVP** [5] foi um dos primeiros a usar camadas de acoplamento afim para garantir a invertibilidade e o cálculo fácil do Jacobiano. O **Glow** [4] aprimorou o RealNVP com convoluções invertíveis 1x1 e Actnorm para melhorar a qualidade da imagem gerada. Os **Continuous Normalizing Flows (CNFs)** [6] modelam o fluxo como uma trajetória contínua no tempo usando Neural ODEs, oferecendo flexibilidade ilimitada de profundidade com eficiência de parâmetros.

## Statistics

**Objetivo de Treinamento:** Maximização da Log-Likelihood Exata (NLL) para todos os modelos. **Paralelização:** Amostragem e Inferência Paralelizáveis (RealNVP, Glow); Amostragem Sequencial, Inferência Paralelizável (CNFs). **Desempenho (Imagens):** Glow alcançou a melhor Log-Likelihood em ImageNet e CelebA entre os modelos de fluxo iniciais. **Custo Computacional:** Variável, sendo o CNF dependente da precisão do ODE Solver, mas geralmente mais eficiente em termos de parâmetros que os modelos discretos. **Invertibilidade:** Garantida por construção em todos os modelos.

## Features

**Normalizing Flows (Geral):** Likelihood exata e tratável; Amostragem e inferência exatas e paralelizadas; Espaço latente significativo e manipulável. **RealNVP:** Uso de camadas de acoplamento afim (affine coupling layers); Arquitetura multi-escala para dados de alta dimensão. **Glow:** Introdução de Convoluções Invertíveis 1x1 para permutar canais; Uso de Actnorm (Activation Normalization); Melhoria na qualidade de geração de imagens. **CNFs:** Uso de Neural ODEs para definir transformações contínuas no tempo; Flexibilidade ilimitada de profundidade (tempo) sem aumento de custo de treinamento; Uso eficiente de parâmetros.

## Use Cases

**Geração de Imagens de Alta Fidelidade:** O Glow é notável por gerar rostos realistas e permitir manipulações semânticas no espaço latente (e.g., mudar a expressão facial) [4]. **Síntese de Voz e Áudio:** Usado em modelos como WaveGlow [7]. **Detecção de Anomalias e Outliers:** A capacidade de calcular a likelihood exata $p(x)$ é ideal para identificar pontos de dados com baixa probabilidade [1]. **Inferência Variacional:** CNFs podem ser usados para inferência variacional mais expressiva e precisa [6]. **Modelagem de Distribuições Físicas e Químicas:** Aplicações em física computacional para modelar distribuições de energia e amostragem eficiente em simulações de Monte Carlo [8]. **Compressão de Dados Lossless:** A modelagem de densidade explícita permite o uso para compressão de dados, onde a log-likelihood se relaciona diretamente com a taxa de bits [1].

## Integration

A integração é tipicamente realizada usando bibliotecas de aprendizado profundo como PyTorch ou TensorFlow. Implementações prontas estão disponíveis em bibliotecas como `nflows` [11]. O treinamento se baseia na maximização da log-likelihood exata. CNFs requerem bibliotecas de solução de Equações Diferenciais Ordinárias (ODEs), como `torchdiffeq`, para resolver a integral do traço do Jacobiano.

**Exemplo Conceitual (PyTorch - Glow-like Flow):**
```python
import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, AffineCouplingTransform, ActNorm, OneByOneConvolution

# 1. Definir a Distribuição Base (Base Distribution)
base_dist = StandardNormal(shape=[2])

# 2. Definir as Transformações (Exemplo de um passo de Glow)
transform = CompositeTransform([
    ActNorm(2),
    OneByOneConvolution(2),
    AffineCouplingTransform(...) # Camada de acoplamento afim
])

# 3. Construir o Flow
flow = Flow(transform, base_dist)

# 4. Treinamento (Maximizar a Log-Likelihood)
# loss = -flow.log_prob(data).mean()
```

## URL

https://arxiv.org/abs/1908.09257