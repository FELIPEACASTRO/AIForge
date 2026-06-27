# Features de Textura (GLCM, LBP, Gabor) para Agropecuária

## Description

O recurso principal é o conjunto de features de textura (GLCM, LBP, Gabor) e sua aplicação em problemas de Agropecuária e Biomassa, conforme demonstrado em estudos recentes (2024-2025). A Matriz de Co-ocorrência de Nível de Cinza (GLCM) é a mais proeminente, fornecendo estatísticas de segunda ordem como Homogeneidade, Contraste, Entropia, Energia, Correlação, Média, Variância, Dissimilaridade e Segundo Momento. O Padrão Binário Local (LBP) e os Filtros de Gabor são usados para capturar informações de textura em diferentes escalas e orientações, sendo frequentemente combinados com técnicas de Deep Learning para melhorar a interpretabilidade e o desempenho em tarefas como classificação de textura do solo e estimativa de biomassa.

## Statistics

**GLCM:** Tipicamente gera um vetor de features de 8 a 14 dimensões por janela de análise (e.g., 3x3, 5x5, 7x7, até 21x21). Em um estudo de caso (2025), a combinação de 8 features GLCM em 4 bandas e 10 tamanhos de janela resultou em 320 features de textura.
**LBP:** O número de features varia com os parâmetros (raio e número de pontos), mas é tipicamente um vetor de histograma de 59 ou 256 dimensões.
**Dataset de Exemplo (2025):** Dataset de Imagens de Textura do Solo com 4.000 imagens rotuladas em 5 classes de textura.

## Features

**GLCM (Gray-Level Co-occurrence Matrix):** Extrai 8 a 14 estatísticas de segunda ordem (e.g., Homogeneidade, Contraste, Entropia) que descrevem a relação espacial entre pixels.
**LBP (Local Binary Patterns):** Descritor de textura robusto e eficiente que captura informações de textura em níveis macro e micro.
**Gabor Filters:** Filtros passa-banda que usam kernels com parâmetros variáveis (gamma, theta, lambda, phi) para realçar padrões específicos de textura e orientação.
**Integração:** Usados como features artesanais (hand-crafted) em frameworks de Machine Learning e Deep Learning (e.g., ATFEM, Random Forest) para complementar features aprendidas.

## Use Cases

**Estimativa de Biomassa:** Usado para estimar a Biomassa Acima do Solo (AGB) em plantações (e.g., borracha) a partir de imagens de sensoriamento remoto (UAV multiespectral).
**Classificação de Textura do Solo:** Classificação de diferentes tipos de textura do solo (e.g., Loamy Sand, Sandy Clay) a partir de imagens de solo.
**Mapeamento de Propriedades Físico-Químicas do Solo:** Estimativa de características como teor de umidade, teor de carbono orgânico e outras propriedades do solo.
**Detecção de Doenças e Estresse em Culturas:** A análise de textura pode identificar mudanças sutis na folhagem ou no solo que indicam estresse ou doença.

## Integration

A extração de features GLCM e LBP é comumente realizada usando bibliotecas de processamento de imagem em Python, como **`scikit-image`** (módulos `feature.graycomatrix` e `feature.local_binary_pattern`) e **`OpenCV`**.
**Exemplo de GLCM (Python - scikit-image):**
```python
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np

# Imagem de entrada (convertida para escala de cinza)
image = rgb2gray(input_image) * 255
image = image.astype(np.uint8)

# Calcular GLCM
glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

# Extrair propriedades (features)
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')
# Outras features como Dissimilarity, ASM (Second Moment), etc.
```
Ferramentas comerciais como **ENVI 5.3** (Co-occurrence Measures) também são utilizadas para extração de GLCM em sensoriamento remoto.

## URL

https://www.nature.com/articles/s41598-025-17384-5