# MS COCO Captions (Microsoft Common Objects in Context Captions)

## Description
O **MS COCO Captions** (Microsoft Common Objects in Context Captions) é um subconjunto do vasto dataset COCO, focado na tarefa de geração de legendas (Image Captioning). Ele é composto por imagens do dataset COCO, cada uma anotada com **cinco legendas distintas** geradas por humanos. O objetivo principal é treinar e avaliar modelos de Visão Computacional e Processamento de Linguagem Natural (NLP) que consigam descrever o conteúdo visual de uma imagem em linguagem natural. É um dos benchmarks mais importantes e amplamente utilizados para a tarefa de *Image Captioning*. A versão mais comum é baseada nas imagens de 2014, mas continua sendo o padrão ouro para a área.

## Statistics
- **Versão Principal:** COCO 2014 Captions (mais utilizada).
- **Imagens:** Aproximadamente 164.000 imagens (82.783 para treino, 40.504 para validação e 40.775 para teste).
- **Legendas:** Mais de 820.000 legendas no total (5 legendas por imagem).
- **Tamanho do Dataset (TFDS):** Download size: 37.61 GiB; Dataset size: 18.83 GiB (incluindo imagens e anotações).
- **Divisão Karpathy:** Uma divisão popular (usada no TFDS) separa o conjunto de validação original em novos conjuntos de treino (`train`: 82.783), validação (`val`: 5.000), teste (`test`: 5.000) e o restante (`restval`: 30.504).

## Features
- **Legendas Múltiplas:** Cada imagem possui 5 legendas independentes geradas por humanos, permitindo uma avaliação mais robusta da qualidade da descrição.
- **Contexto Rico:** As imagens são complexas e contêm múltiplos objetos em contextos cotidianos, o que exige que os modelos compreendam as interações entre os objetos.
- **Integração com Outras Tarefas COCO:** O dataset de legendas utiliza as mesmas imagens do COCO, permitindo o uso de anotações de detecção de objetos e segmentação para tarefas multimodais.
- **Avaliação Padronizada:** O dataset é acompanhado por um conjunto de métricas de avaliação (BLEU, METEOR, ROUGE-L, CIDEr, SPICE) e um servidor de avaliação para comparação de desempenho.

## Use Cases
- **Geração Automática de Legendas (Image Captioning):** Treinamento e avaliação de modelos que geram descrições textuais para imagens.
- **Visão e Linguagem Multimodal:** Pesquisa em modelos que integram a compreensão visual e a geração de linguagem natural.
- **Recuperação de Imagens Baseada em Texto:** Desenvolvimento de sistemas que buscam imagens com base em consultas textuais.
- **Modelos de Atenção Visual:** Estudo de como os modelos focam em partes específicas da imagem ao gerar palavras da legenda.
- **Transferência de Aprendizado:** Pré-treinamento de *encoders* visuais e *decoders* de linguagem para tarefas relacionadas.

## Integration
O dataset pode ser acessado de várias formas, sendo a mais comum através da API oficial do COCO ou de bibliotecas de terceiros:

1.  **API Oficial (COCO API):** A maneira mais direta é usar a [COCO API](https://github.com/cocodataset/cocoapi), que fornece ferramentas para carregar, analisar e visualizar as anotações.
2.  **TensorFlow Datasets (TFDS):** Para usuários de TensorFlow, o dataset está disponível como `coco_captions` no TFDS, facilitando o download e a integração com pipelines de treinamento:
    ```python
    import tensorflow_datasets as tfds
    ds = tfds.load('coco_captions', split='train', shuffle_files=True)
    ```
3.  **Hugging Face Datasets:** O dataset também pode ser encontrado no Hugging Face Hub, simplificando o uso com a biblioteca `datasets` para modelos de NLP/Visão:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("coco_captions")
    ```
A versão 2014 é a mais utilizada para *captioning*. É necessário baixar as imagens (COCO 2014 Images) e os arquivos de anotação (Captions Annotations).

## URL
[https://cocodataset.org/#captions-2015](https://cocodataset.org/#captions-2015)
