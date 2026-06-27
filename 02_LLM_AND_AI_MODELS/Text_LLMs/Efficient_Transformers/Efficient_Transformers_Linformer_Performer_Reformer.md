# Efficient Transformers: Linformer, Performer, Reformer

## Description

O Linformer, Performer e Reformer são arquiteturas de Transformer eficientes projetadas para superar a limitação de complexidade quadrática (O(n²)) do mecanismo de auto-atenção padrão, permitindo o processamento de sequências de dados significativamente mais longas com maior eficiência de tempo e memória. O **Linformer** alcança a complexidade linear (O(n)) ao aproximar a auto-atenção por uma matriz de baixo posto. O **Performer** também atinge a complexidade linear (O(n)) através do algoritmo FAVOR+ (Fast Attention Via Positive Orthogonal Random Features), que fornece uma estimativa imparcial da atenção softmax. Já o **Reformer** utiliza Locality-Sensitive Hashing (LSH) para reduzir a complexidade para O(n log n) e Camadas Residuais Reversíveis para otimizar drasticamente o uso de memória, permitindo janelas de contexto de até 1 milhão de palavras. Todos são cruciais para aplicações de modelagem de sequências longas em NLP, genômica e visão computacional.

## Statistics

Linformer: Complexidade O(n) em tempo e espaço. Performer: Complexidade O(n) em tempo e espaço. Reformer: Complexidade O(n log n) na atenção, Janela de Contexto de até 1 milhão de palavras, Uso de Memória de 16 GB em um único acelerador.

## Features

Linformer: Atenção Linear (O(n)), Aproximação por Matriz de Baixo Posto, Eficiência de Memória e Tempo. Performer: Atenção Linear (O(n)), Algoritmo FAVOR+, Atenção Baseada em Kernel, Compatibilidade com Pré-treinamento. Reformer: Atenção LSH (O(n log n)), Camadas Residuais Reversíveis, Chunking, Eficiência Extrema para Sequências Longas.

## Use Cases

Modelagem de Sequências Longas (NLP, Genômica); Classificação de Texto (Linformer); Geração de Imagens e Vídeos (Reformer); Aplicações em Vision Transformers (Linformer); Compreensão de Contexto em Larga Escala.

## Integration

Linformer e Performer são integrados principalmente via bibliotecas PyTorch de terceiros (ex: lucidrains). O Reformer possui integração oficial e nativa na biblioteca Hugging Face Transformers, facilitando o uso de modelos pré-treinados. Todos oferecem exemplos de código conceitual em Python/PyTorch para implementação de seus mecanismos de atenção e arquiteturas.

## URL

https://arxiv.org/abs/2006.04768 (Linformer), https://research.google/blog/rethinking-attention-with-performers/ (Performer), https://research.google/blog/reformer-the-efficient-transformer/ (Reformer)