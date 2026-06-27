# AgriTransformer

## Description

AgriTransformer é uma arquitetura de deep learning baseada em Transformer que utiliza mecanismos de co-atenção para integrar dados multimodais (dados agrícolas tabulares e índices de vegetação de satélites Sentinel-2) para aprimorar a previsão de rendimento de culturas. O modelo é projetado para capturar dependências dinâmicas entre as variáveis ambientais e fisiológicas das plantas.

## Statistics

Coeficiente de determinação (R2) de 0,919 e Erro Quadrático Médio (MSE) de 2,598 para a variante com co-atenção. Supera significativamente modelos tradicionais de regressão linear. Publicado em Electronics (2025).

## Features

Mecanismo de co-atenção adaptado ao domínio agrícola, integração de fontes de dados heterogêneas (tabulares e satelitais), modularidade para adaptação a diferentes culturas e regiões, alta robustez e interpretabilidade.

## Use Cases

Agricultura de precisão, planejamento de colheitas, gerenciamento de recursos hídricos e de nutrientes, e previsão de produtividade agrícola em geral.

## Integration

Implementação detalhada usando Google Colaboratory, TensorFlow e Keras. Scripts em Python para obtenção de imagens satelitais via Google Earth Engine. Código disponível no GitHub: https://github.com/lrobertojacomeg/multimodal

## URL

https://www.mdpi.com/2079-9292/14/12/2466