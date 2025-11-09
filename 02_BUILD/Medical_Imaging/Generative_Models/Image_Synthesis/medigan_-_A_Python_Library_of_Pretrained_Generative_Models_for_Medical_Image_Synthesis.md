# medigan - A Python Library of Pretrained Generative Models for Medical Image Synthesis

## Description

Biblioteca Python que fornece síntese de imagens médicas de forma amigável, permitindo aos usuários escolher entre uma variedade de modelos generativos pré-treinados (GANs, CycleGAN, Pix2Pix, etc.) para gerar conjuntos de dados sintéticos. É uma solução para a escassez de dados em imagens médicas, facilitando o compartilhamento de dados e o aumento de dados (data augmentation).

## Statistics

O repositório GitHub possui 189 estrelas e 21 forks. O artigo de referência (medigan paper) contém análise e comparações de modelos e pontuações FID (Frechet Inception Distance).

## Features

Solução para escassez de dados em imagens médicas; Compartilhamento de dados via modelos generativos; Aumento de dados (data augmentation); Adaptação de domínio; Mais de 20 modelos disponíveis para diversas modalidades (mamografia, ressonância magnética cerebral, endoscopia, raio-x de tórax).

## Use Cases

Treinamento ou adaptação de modelos de IA para tarefas clínicas (classificação de lesões, segmentação, detecção); Geração de massas mamárias malignas/benignas com labels; Transferência de densidade mamária (Low-to-High Density) usando CycleGAN.

## Integration

```python
from medigan import Generators
generators = Generators()
# Gera 8 amostras com o modelo 8 (00008_C-DCGAN_MMG_MASSES).
generators.generate(model_id=8, num_samples=8, install_dependencies=True)
```

## URL

https://github.com/RichardObi/medigan