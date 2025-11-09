# Conceptual Captions (CC3M e CC12M)

## Description
O Conceptual Captions é uma família de datasets de grande escala, desenvolvida pelo Google, que consiste em pares de (URL da imagem, legenda) coletados da web. A versão original, CC3M, contém aproximadamente 3,3 milhões de pares, com legendas extraídas do atributo Alt-text do HTML e processadas por um pipeline automático para garantir limpeza e informatividade. A versão mais recente e maior, Conceptual 12M (CC12M), possui cerca de 12 milhões de pares e foi especificamente projetada para o pré-treinamento de modelos de visão e linguagem em larga escala, cobrindo um conjunto mais diversificado de conceitos visuais. O CC12M é intencionalmente disjunto do CC3M, embora uma pequena intersecção de URLs exista.

## Statistics
CC3M: Aproximadamente 3,3 milhões de pares imagem-legenda. CC12M: Aproximadamente 12 milhões de pares imagem-texto. O CC12M é distribuído em um arquivo TSV de 2.5GB (apenas URLs e legendas).

## Features
Legendas 'conceituais' e não descritivas, extraídas de forma automática da web, resultando em um estilo mais variado e menos curado em comparação com datasets como o MS-COCO. Foco em pré-treinamento de modelos de Visão e Linguagem (V&L). O CC12M visa reconhecer conceitos visuais de 'cauda longa' (long-tail).

## Use Cases
Pré-treinamento de modelos de Visão e Linguagem (V&L), como CLIP e modelos de Geração de Legendas de Imagem (Image Captioning). Benchmarking para tarefas de compreensão visual e geração de texto.

## Integration
O dataset é fornecido como um arquivo TSV (Tab-Separated Values) contendo URLs de imagens e suas respectivas legendas. Os usuários devem baixar o arquivo TSV e, em seguida, usar as URLs para baixar as imagens. O Google não fornece as imagens diretamente devido a questões de direitos autorais e a natureza dinâmica da web. O arquivo CC12M (URLs e legendas) tem cerca de 2.5GB. Versões do dataset estão disponíveis em plataformas como Hugging Face e Kaggle, mas a fonte oficial é o Google Research.

## URL
[https://ai.google.com/research/ConceptualCaptions](https://ai.google.com/research/ConceptualCaptions)
