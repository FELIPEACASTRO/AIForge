# RaTE-NER

## Description

Um dataset de grande escala para Reconhecimento de Entidades Nomeadas (NER) em radiologia, abrangendo 9 modalidades de imagem e 23 regiões anatômicas. O dataset foi construído a partir de 13.235 sentenças anotadas manualmente de 1.816 relatórios do banco de dados MIMIC-IV, e enriquecido com 33.605 sentenças de 17.432 relatórios do Radiopaedia, com anotação inicial assistida por GPT-4.

## Statistics

Total de 13.235 sentenças anotadas manualmente (MIMIC-IV) e 33.605 sentenças enriquecidas (Radiopaedia). Total de 1.816 relatórios do MIMIC-IV e 17.432 relatórios do Radiopaedia. O conjunto de teste (test set) possui 3.529 sentenças rotuladas manualmente.

## Features

Foco em entidades radiológicas. Suporta 9 modalidades de imagem e 23 regiões anatômicas. Oferece dois formatos de pré-processamento: IOB (Inside, Outside, Beginning) e span tagging.

## Use Cases

Treinamento e avaliação de modelos de NER para extração de informações estruturadas de relatórios radiológicos. Aplicações em sistemas de suporte à decisão clínica e pesquisa em IA médica.

## Integration

Pode ser acessado diretamente via Hugging Face Datasets: `from datasets import load_dataset; data = load_dataset("Angelakeke/RaTE-NER")`

## URL

https://huggingface.co/datasets/Angelakeke/RaTE-NER