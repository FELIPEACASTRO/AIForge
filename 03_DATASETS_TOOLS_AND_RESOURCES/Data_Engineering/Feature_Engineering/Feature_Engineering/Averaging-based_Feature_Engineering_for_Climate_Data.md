# Averaging-based Feature Engineering for Climate Data

## Description

Técnica de Engenharia de Features baseada em Média (Averaging-based Feature Engineering) para dados climáticos, como Temperatura, Precipitação e Umidade. Esta técnica é utilizada para capturar os efeitos de defasagem temporal (time-lagging effects) das variáveis climáticas na predição de rendimento de pastagem e, por extensão, na estimativa de biomassa. A abordagem calcula a média de um atributo climático (ex: precipitação) ao longo de múltiplos períodos de tempo (ex: 3, 5, 7, 10, 14, 30, 90, 180 e 365 dias), transformando um único atributo em múltiplos atributos que resumem seu comportamento temporal histórico. Isso permite que modelos de Machine Learning, como o XGBoost, aprendam padrões complexos de longo e curto prazo, melhorando significativamente a precisão da predição.

## Statistics

O estudo que descreve a técnica utilizou dados de 196 fazendas (e 6885 piquetes) na Austrália, cobrindo dois anos (2019 e 2020). Os 7 atributos climáticos originais foram expandidos para 70 atributos (7 originais + 63 features de média). A técnica resultou em melhorias de até 31.81% nas métricas de avaliação (RMSE, MAE, R²) para a predição de rendimento de pastagem.

## Features

A técnica transforma 7 atributos climáticos originais (incluindo Precipitação, Temperatura Máxima/Mínima, Umidade, Evaporação, Radiação Solar e Pressão de Vapor) em 63 novos atributos (7 atributos x 9 períodos de tempo). Os períodos de média são: 3, 5, 7, 10, 14, 30, 90, 180 e 365 dias. Os atributos resultantes capturam tendências de curto, médio e longo prazo, sendo cruciais para modelar a resposta biológica das plantas às condições climáticas acumuladas.

## Use Cases

Predição de rendimento de pastagem (pasture yield prediction), que é um proxy direto para a estimativa de biomassa em sistemas agrícolas. O método é aplicável a qualquer tarefa de Machine Learning em Agricultura de Precisão que envolva a modelagem da influência de longo prazo de variáveis climáticas (Temperatura, Precipitação, Umidade) em variáveis biológicas (crescimento de culturas, biomassa, produtividade).

## Integration

A integração é feita por meio de um pré-processamento de dados onde o valor médio de cada atributo climático é calculado para os períodos de tempo definidos (3 a 365 dias) antes de alimentar o modelo de Machine Learning. O artigo sugere o uso de modelos como XGBoost, Random Forest e Redes Neurais para aproveitar esses recursos. Embora o código específico não esteja disponível, a lógica de implementação envolve a aplicação de uma janela deslizante de média (rolling average) sobre a série temporal dos dados climáticos.

## URL

https://www.sciencedirect.com/science/article/pii/S1574954125000202