# Integração de Imagem Hiperespectral e IA para Mapeamento e Saúde de Culturas

## Description

Revisão sistemática e análise de tendências recentes (2023-2025) na aplicação de Imagem Hiperespectral (HSI) combinada com modelos avançados de Inteligência Artificial (IA), como Deep Learning (DL) e Vision Transformers (ViTs), para a agricultura de precisão. O foco principal é a discriminação precisa de tipos de culturas e a avaliação da saúde das plantas, utilizando dados de sensores aéreos (UAV) e espaciais (e.g., EnMAP, PRISMA). A pesquisa destaca a evolução para modelos mais complexos e escaláveis, como as Redes Neurais Gráficas (GNNs) e os Modelos Fundamentais Geoespaciais (GFMs), como a próxima fronteira para o monitoramento agrícola em larga escala.

## Statistics

A pesquisa recente (2023-2025) aponta para o uso crescente de Vision Transformers (ViTs) e arquiteturas híbridas, que demonstram alta precisão de classificação em comparação com métodos tradicionais de Machine Learning. O campo está evoluindo rapidamente, com artigos de revisão de 2025 já citando a necessidade de adoção de Geospatial Foundation Models (GFMs) e Graph Neural Networks (GNNs) para lidar com a alta dimensionalidade dos dados HSI e permitir o mapeamento de culturas em grandes áreas. Exemplos específicos em trigo incluem o uso de Deep Learning com Mecanismo de Atenção e estratégias de Transferência para estimativa de produtividade (2024), e Machine Learning para análise de grãos danificados (2023).

## Features

Capacidade de discriminação precisa de tipos de culturas; avaliação não destrutiva e em tempo real de parâmetros biofísicos (e.g., teor de nitrogênio, clorofila); detecção precoce de estresse hídrico e doenças (e.g., *Fusarium head blight*); estimativa de produtividade e biomassa; uso de dados de alta resolução espectral (centenas de bandas) de plataformas UAV e satélites.

## Use Cases

Mapeamento de tipos de culturas em escala regional e global para segurança alimentar; monitoramento da saúde das culturas e diagnóstico de doenças em estágios iniciais; otimização da aplicação de fertilizantes (nitrogênio) e irrigação através da estimativa de nutrientes e estresse; fenotipagem de alto rendimento em programas de melhoramento genético; estimativa de produtividade e biomassa para gestão agrícola e seguros.

## Integration

A integração é tipicamente realizada através de pipelines de processamento de dados que envolvem: 1. Pré-processamento de dados HSI (correção radiométrica e atmosférica); 2. Redução de dimensionalidade (e.g., PCA, seleção de bandas); 3. Treinamento de modelos de Deep Learning (e.g., ViTs, CNNs) ou Machine Learning (e.g., SVM, Random Forest) para classificação ou regressão. O repositório GitHub `fadi-07/Awesome-Wheat-HSI-DeepLearning` serve como um recurso de referência para artigos e implementações em trigo, indicando a base para o desenvolvimento de código em Python com bibliotecas como PyTorch/TensorFlow para os modelos de DL e scikit-learn para ML.

## URL

https://www.mdpi.com/2072-4292/17/9/1574