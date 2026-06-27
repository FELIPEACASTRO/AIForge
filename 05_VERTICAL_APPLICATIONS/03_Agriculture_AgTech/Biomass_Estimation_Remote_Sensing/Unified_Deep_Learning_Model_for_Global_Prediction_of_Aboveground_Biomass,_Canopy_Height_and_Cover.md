# Unified Deep Learning Model for Global Prediction of Aboveground Biomass, Canopy Height and Cover

## Description

Um modelo de aprendizado profundo unificado que realiza a previsão da Densidade de Biomassa Acima do Solo (AGBD), Altura do Dossel (CH) e Cobertura do Dossel (CC), juntamente com estimativas de incerteza para todas as três quantidades. Utiliza imagens multissensor e multiespectrais de alta resolução (10 metros) e é treinado em milhões de medições GEDI-L2/L4 amostradas globalmente.

## Statistics

Treinado em milhões de medições GEDI-L2/L4 amostradas globalmente. Atinge um Erro Absoluto Médio (MAE) para AGBD (CH, CC) de 26,1 Mg/ha (3,7 m, 9,9 %) e um Erro Quadrático Médio da Raiz (RMSE) de 50,6 Mg/ha (5,4 m, 15,8 %) em um conjunto de dados de teste amostrado globalmente. Validado globalmente para 2023 e anualmente de 2016 a 2023 em áreas selecionadas. Demonstra melhoria significativa em relação a resultados publicados anteriormente.

## Features

Previsão unificada de AGBD, CH e CC. Fornece estimativas de incerteza. Utiliza imagens de satélite multissensor e multiespectrais de alta resolução (10m). Arquitetura multi-cabeça que facilita a transferibilidade para outras variáveis GEDI.

## Use Cases

Medição regular do estoque de carbono nas florestas mundiais. Contabilidade e relatórios de carbono sob iniciativas climáticas nacionais e internacionais. Pesquisa científica sobre ecossistemas florestais.

## Integration

O resumo menciona um modelo pré-treinado e uma arquitetura multi-cabeça que facilita a transferibilidade, sugerindo potencial para uso. O artigo é do arXiv, portanto, um link direto para o código ou repositório GitHub não está imediatamente disponível, mas o modelo foi implantado globalmente para 2023 pelos autores.

## URL

https://arxiv.org/abs/2408.11234