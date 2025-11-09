# Global Wheat Head Detection (GWHD) Dataset e Global Wheat Full Semantic Organ Segmentation (GWFSS) Dataset

## Description

O **Global Wheat Head Detection (GWHD) Dataset** é um dos maiores e mais diversos conjuntos de dados de imagens de trigo rotuladas em alta resolução, criado por um consórcio internacional de 11 instituições em 7 países. O objetivo principal é desenvolver e comparar métodos de visão computacional para a detecção e contagem de espigas de trigo (wheat heads) em condições de campo.

A versão original (GWHD 2020) foi expandida para a **GWHD 2021**, que melhorou a diversidade e a confiabilidade dos rótulos. Mais recentemente, o artigo de 2025 introduziu o **Global Wheat Full Semantic Organ Segmentation (GWFSS) dataset**, uma evolução que visa a segmentação semântica completa dos órgãos da planta (folhas, caules e espigas), cobrindo todos os estágios de desenvolvimento, o que representa um avanço significativo em relação à detecção de caixas delimitadoras (bounding boxes) do GWHD.

O dataset é crucial para o avanço da fenotipagem de alto rendimento, permitindo a estimativa de traços importantes como a densidade populacional de espigas, que é um componente chave do rendimento do trigo. A diversidade de genótipos, ambientes e condições de aquisição de imagem torna o GWHD um benchmark robusto para modelos de aprendizado profundo.

## Statistics

- **GWHD 2020 (Original):** 4.700 imagens RGB de alta resolução e aproximadamente 190.000 espigas de trigo rotuladas.
- **GWHD 2021 (Atualização):** 6.515 imagens RGB de alta resolução, representando 275.187 espigas de trigo rotuladas.
- **GWFSS (2025 - Evolução):** 1.096 imagens diversas com segmentação semântica de órgãos em nível de pixel (folhas, caules e espigas) e um conjunto adicional de 52.078 imagens sem anotações para pré-treinamento.
- **Tamanho do Arquivo (Kaggle):** A versão da competição Kaggle tem um tamanho de 643.57 MB (3434 arquivos, incluindo imagens e CSVs).
- **Densidade de Rótulos:** Média de 40 espigas por imagem, com uma distribuição ampla.
- **Resolução:** Imagens harmonizadas para uma resolução efetiva de 0.21 a 0.55 mm/pixel.

## Features

- **Diversidade Global**: Imagens coletadas em 7 países (Japão, França, Canadá, Reino Unido, Suíça, China, Austrália) e 10 locais, cobrindo uma ampla gama de genótipos, condições ambientais e estágios de desenvolvimento.
- **Alta Resolução**: Imagens RGB de alta resolução (originalmente até 6000x4000 pixels) que foram harmonizadas e divididas em patches de 1024x1024 pixels.
- **Rótulos de Caixa Delimitadora (Bounding Box)**: O dataset GWHD original fornece caixas delimitadoras para cada espiga de trigo identificada.
- **Evolução para Segmentação Semântica (GWFSS)**: A evolução mais recente (GWFSS, 2025) fornece segmentação em nível de pixel para órgãos completos (folhas, caules e espigas), incluindo tecidos necróticos e senescentes, o que é fundamental para a classificação de tecidos saudáveis vs. não saudáveis.
- **Desafio de Generalização**: O dataset foi dividido intencionalmente para testar a generalização, com o conjunto de treinamento focado na Europa/América do Norte e o conjunto de teste na Austrália/Japão/China.

## Use Cases

- **Detecção e Contagem de Espigas de Trigo (Wheat Head Detection and Counting):** O caso de uso primário, crucial para estimar a densidade populacional de espigas, um componente chave do rendimento.
- **Fenotipagem de Alto Rendimento (High-Throughput Phenotyping):** Extração de traços fenotípicos adicionais, como tamanho, inclinação, cor, estágio de maturação e saúde das espigas.
- **Desenvolvimento de Modelos de Visão Computacional:** Benchmark para o desenvolvimento e avaliação de modelos de aprendizado profundo (e.g., YOLO, Faster R-CNN, Segformer) em tarefas de detecção de objetos e segmentação semântica em ambientes agrícolas complexos.
- **Estudo de Generalização de Modelos:** O dataset é ideal para testar a robustez de modelos em condições não vistas (diferentes genótipos, ambientes e câmeras).
- **Segmentação Semântica de Órgãos (GWFSS):** Aplicação avançada para a classificação de tecidos saudáveis vs. não saudáveis, quantificação de senescência e doenças em dosséis de trigo.

## Integration

O dataset é acessível publicamente e foi a base para as competições Kaggle "Global Wheat Detection" (2020 e 2021).

**Acesso ao Dataset:**
1.  **Website Oficial:** O dataset está disponível para download e mais informações em [http://www.global-wheat.com/](http://www.global-wheat.com/).
2.  **Kaggle:** Os dados originais da competição estão disponíveis no Kaggle, embora a competição em si esteja encerrada: [https://www.kaggle.com/competitions/global-wheat-detection/data](https://www.kaggle.com/competitions/global-wheat-detection/data).

**Exemplo de Integração (Conceitual - Python/PyTorch/TensorFlow):**
A integração tipicamente envolve o download dos arquivos de imagem (`train.zip`, `test.zip`) e do arquivo de rótulos (`train.csv`). O `train.csv` contém as caixas delimitadoras no formato `[xmin, ymin, width, height]`.

```python
import pandas as pd
import os

# Carregar o arquivo de rótulos
labels_df = pd.read_csv('train.csv')

# Exemplo de estrutura do DataFrame
# image_id, width, height, bbox
# 8c148c04c, 1024, 1024, [10, 20, 50, 60]

def parse_bbox(bbox_str):
    # Converte a string do bbox para uma lista de inteiros
    # Ex: "[10, 20, 50, 60]" -> [10, 20, 50, 60]
    import ast
    return ast.literal_eval(bbox_str)

# Aplicar a função para obter as coordenadas
labels_df['bbox_coords'] = labels_df['bbox'].apply(parse_bbox)

# Exemplo de como carregar uma imagem e suas caixas delimitadoras
image_id = labels_df['image_id'].iloc[0]
image_path = os.path.join('train', f'{image_id}.jpg')

# Em um pipeline de Deep Learning (e.g., com PyTorch ou TensorFlow),
# o próximo passo seria carregar a imagem, aplicar transformações
# e formatar os rótulos para o formato esperado pelo modelo (e.g., YOLO, Faster R-CNN).
# As coordenadas do bbox seriam usadas para treinar o modelo de detecção de objetos.
```

## URL

http://www.global-wheat.com/