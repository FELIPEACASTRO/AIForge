# Open Images Dataset V7 (Google)

## Description
O Open Images Dataset V7 é um dos maiores e mais diversos conjuntos de dados de visão computacional, contendo aproximadamente 9 milhões de imagens anotadas com uma variedade de rótulos. Ele suporta múltiplas tarefas, incluindo classificação de imagens, detecção de objetos, segmentação de instâncias, detecção de relações visuais e narrativas localizadas. A versão V7, lançada em Outubro de 2022, adicionou 66.4 milhões de rótulos de nível de ponto (point-level labels) em 5.827 classes, tornando-o adequado para treinamento e avaliação de segmentação semântica zero/few-shot. O dataset é conhecido pela sua diversidade e complexidade de cenas, com uma média de 8.3 objetos por imagem.

## Statistics
**Versão Mais Recente:** V7 (Outubro de 2022). **Total de Imagens:** ~9 milhões. **Amostras Densa e Anotadas:** 1.9 milhões de imagens com anotações densas. **Bounding Boxes:** 16 milhões em 600 classes. **Segmentações de Instância:** 2.8 milhões em 350 classes. **Relações Visuais:** 3.3 milhões em 1,466 relações. **Narrativas Localizadas:** 675 mil. **Rótulos de Nível de Ponto (V7):** 66.4 milhões em 5,827 classes. **Rótulos de Nível de Imagem:** 61.4 milhões em 20,638 classes.

## Features
Suporte a múltiplas tarefas de Visão Computacional: Classificação de Imagens, Detecção de Objetos (Bounding Boxes), Segmentação de Instâncias (Masks), Detecção de Relações Visuais e Narrativas Localizadas. A V7 introduziu 66.4M de rótulos de nível de ponto (point-level labels) para segmentação semântica. Anotações de alta qualidade, muitas delas verificadas manualmente por anotadores profissionais. Grande diversidade de imagens e complexidade de cenas.

## Use Cases
Treinamento e avaliação de modelos de detecção de objetos (object detection), segmentação de instâncias (instance segmentation), classificação de imagens multi-rótulo (multi-label image classification), detecção de relações visuais (visual relationship detection) e tarefas multimodais que combinam visão e linguagem (localized narratives). É frequentemente usado como um benchmark para modelos de visão computacional em larga escala.

## Integration
O dataset pode ser acessado e baixado de três maneiras principais: 1. **Download Manual:** Arquivos de anotação e metadados em formato CSV e um script Python para download das imagens. 2. **TensorFlow Datasets (TFDS):** Acesso via `tfds.load('open_images/v7')`. 3. **FiftyOne:** Utilização da biblioteca FiftyOne para download e visualização de subconjuntos específicos do dataset, como `fiftyone.zoo.load_zoo_dataset("open-images-v7")`.

## URL
[https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
