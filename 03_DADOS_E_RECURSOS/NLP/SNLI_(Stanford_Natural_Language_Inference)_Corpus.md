# SNLI (Stanford Natural Language Inference) Corpus

## Description
O Corpus SNLI (versão 1.0) é uma coleção de 570 mil pares de sentenças em inglês, escritos por humanos e rotulados manualmente para classificação balanceada com os rótulos **entailment** (implicação), **contradiction** (contradição) e **neutral** (neutro). O objetivo principal é servir como um *benchmark* para avaliar sistemas de representação de texto, especialmente aqueles induzidos por métodos de aprendizado de representação, e como um recurso para o desenvolvimento de modelos de Processamento de Linguagem Natural (PLN).

## Statistics
Tamanho do dataset: 20.4 MB (arquivos Parquet). Contagem de amostras: 570.152 pares de sentenças no total. Versão: 1.0. Divisões: Treinamento (550.152), Validação (10.000), Teste (10.000).

## Features
Tarefa de Inferência de Linguagem Natural (NLI) ou Reconhecimento de Implicação Textual (RTE). Rótulos: Implicação, Contradição e Neutro. Pares de sentenças (Premissa e Hipótese). Hipóteses geradas por *crowdworkers* com base em legendas de imagens (Flickr 30k e VisualGenome). Alta qualidade de anotação com 98% de concordância de 3/5 anotadores no conjunto de validação.

## Use Cases
Treinamento e avaliação de modelos de PLN para Inferência de Linguagem Natural. Desenvolvimento de sistemas de representação de texto. *Benchmark* para modelos de aprendizado profundo (Deep Learning) em tarefas de compreensão de linguagem. Base para a criação de datasets estendidos, como o e-SNLI (com explicações em linguagem natural).

## Integration
O dataset pode ser baixado diretamente do site oficial de Stanford (arquivo zip) ou acessado facilmente através de bibliotecas de datasets de PLN. \n\n**Hugging Face Datasets (Python):**\n```python\nfrom datasets import load_dataset\ndataset = load_dataset(\"stanfordnlp/snli\")\n```\n\n**TensorFlow Datasets (Python):**\n```python\nimport tensorflow_datasets as tfds\ndataset, info = tfds.load('snli', with_info=True)\n```

## URL
[https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
