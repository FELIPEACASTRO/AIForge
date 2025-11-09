# CSpace: a concept embedding space for biomedical applications

## Description

CSpace é um espaço de *word embedding* conciso de conceitos biomédicos que supera alternativas em termos de taxa de *out-of-vocabulary* (OOV) e similaridade textual semântica (STS). Ele também codifica IDs ontológicos (MeSH, NCBI gene e tax ID), permitindo a medição da relação entre doenças, genes ou condições e a busca semântica de sinônimos. Foi desenvolvido pelo cosbi-research.

## Statistics

O modelo CSpace 2024 (última versão em 2 de abril de 2025) foi treinado com dados até agosto de 2024. O repositório contém um exemplo de 81K publicações pré-processadas para reconstrução do conjunto de treinamento. Os embeddings fine-tuned estão disponíveis no Zenodo.

## Features

Word embedding conciso de conceitos biomédicos. Baixa taxa OOV. Alta performance em STS. Codifica IDs ontológicos (MeSH, NCBI gene e tax ID). Utiliza Python e R.

## Use Cases

Medição da relação entre doenças, genes ou condições. Descoberta de associações desconhecidas entre doença e condição. Busca semântica de sinônimos. Tarefas de similaridade de sentenças (BIOSSES).

## Integration

A integração é feita via Python. É necessário instalar as dependências (`pip install -r requirements.txt`) e utilizar os scripts de exemplo (`example.py`, `test.py`, `test_sentence.py`) com os arquivos binários dos embeddings (`cspace.kv.bin`, `cspace.bigrams.pkl`, `cspace.dict.pkl`) obtidos no Zenodo.

## URL

https://github.com/cosbi-research/cspace