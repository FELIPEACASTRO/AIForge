# PlantVillage Dataset (Plant Disease)

## Description

O PlantVillage Dataset é uma coleção de imagens de folhas de plantas saudáveis e doentes, sendo um dos benchmarks mais utilizados para o desenvolvimento de modelos de Deep Learning na detecção de doenças agrícolas. A versão mais referenciada e utilizada em pesquisas recentes (2023-2025) consiste em 53.606 imagens de 14 espécies de culturas, divididas em 38 classes (saudáveis e doentes). O dataset é crucial para o avanço da agricultura de precisão, permitindo o treinamento de sistemas de diagnóstico rápido e preciso, inclusive para aplicações em dispositivos de borda (edge computing). Pesquisas recentes focam em técnicas avançadas de extração de features e modelos híbridos para melhorar a acurácia e reduzir a complexidade computacional.

## Statistics

**Total de Imagens:** 53.606
**Dimensão das Imagens:** 256 x 256
**Total de Espécies de Plantas:** 14
**Total de Classes (Saudáveis e Doentes):** 38

**Distribuição de Classes (Exemplos):**
- **Tomato:** 10 classes (18.160 imagens)
- **Apple:** 4 classes (3.171 imagens)
- **Orange:** 1 classe (5.507 imagens)

**Acurácia de Referência (Artigo 2025):** 98.95% (com modelo híbrido ResNet-KELM).

## Features

Imagens de folhas de 14 espécies de culturas (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato). 38 classes distintas, incluindo estados saudáveis e 26 tipos de doenças. Imagens padronizadas com dimensão de 256x256 pixels. Utilizado como base para técnicas de Deep Feature Extraction (ResNet-50, DenseNet, EfficientNet) e modelos híbridos (ResNet-KELM).

## Use Cases

**Detecção e Classificação de Doenças:** Identificação de doenças foliares em culturas.
**Desenvolvimento de Aplicações Móveis:** Criação de ferramentas de diagnóstico em tempo real para agricultores (edge computing).
**Pesquisa e Benchmarking:** Avaliação de novas arquiteturas de Deep Learning e técnicas de feature engineering (e.g., ResNet-KELM, Fusão de Features, Mecanismos de Atenção).
**Transfer Learning:** Pré-treinamento de modelos para tarefas de classificação de doenças.

## Integration

O dataset é facilmente acessível via repositórios como Kaggle e TensorFlow Datasets (TFDS).

**Exemplo de Acesso (Python/TFDS):**
```python
import tensorflow_datasets as tfds

# Carregar o dataset PlantVillage
ds, ds_info = tfds.load('plant_village', split='train', with_info=True)

# Exibir informações do dataset
print(ds_info)

# O dataset é frequentemente usado para Transfer Learning com modelos pré-treinados.
```

## URL

https://www.kaggle.com/datasets/emmarex/plantdisease (Kaggle) ou https://www.tensorflow.org/datasets/catalog/plant_village (TFDS)