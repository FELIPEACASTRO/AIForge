# CLMS (Continual Learning Multi-Scale domain adaptation)

## Description

**CLMS (Continual Learning Multi-Scale domain adaptation)** é uma estrutura de ponta para a adaptação de modelos de Deep Learning em novos domínios clínicos de imagens médicas, mesmo quando os dados originais de treinamento (fonte) não estão disponíveis (Source-Free Domain Adaptation - SFDA). O CLMS integra reconstrução multi-escala, aprendizagem contínua (CL) e alinhamento de estilo para superar desafios como a propagação de erros, o desalinhamento de características visuais e estruturais, e o esquecimento catastrófico do conhecimento prévio. O objetivo é garantir a implantação segura e confiável de modelos de IA em diversos ambientes de saúde.

## Statistics

- **Segmentação de Ressonância Magnética da Próstata:** Melhoria de **10,87%** no coeficiente Dice (em comparação com o estado da arte).
- **Segmentação de Pólipos em Colonoscopia:** Melhoria de **17,73%** no coeficiente Dice.
- **Classificação de Doença Plus em Imagens de Retina:** Melhoria de **11,19%** na AUC (Area Under the Curve).
- **Preservação de Conhecimento:** Preservou o conhecimento da fonte para todas as tarefas, evitando o esquecimento catastrófico.
- **Citações:** Citado por 8 (em dezembro de 2024, conforme snippet).
- **Publicação:** *Medical Image Analysis* (2025).

## Features

- **Adaptação de Domínio Livre de Fonte (SFDA):** Adapta modelos a novos domínios de destino não rotulados sem acesso aos dados de origem originais.
- **Aprendizagem Contínua (CL):** Preserva o conhecimento da fonte e evita o esquecimento catastrófico ao integrar novos dados de domínio de destino.
- **Reconstrução Multi-Escala:** Aborda o desalinhamento de características visuais e estruturais entre domínios.
- **Alinhamento de Estilo:** Ajuda a reduzir as discrepâncias de dados entre diferentes locais clínicos.
- **Solução End-to-End:** Estrutura completa para transferência e adaptação robusta de conhecimento.

## Use Cases

- **Implantação de Modelos em Novos Hospitais/Clínicas:** Adaptação de modelos de IA treinados em um local para uso em outro, onde as características dos dados (equipamento, população) são diferentes.
- **Segmentação Robusta de Imagens Médicas:** Aplicação em tarefas críticas como segmentação de tumores, órgãos ou lesões em diferentes modalidades de imagem (MRI, Colonoscopia, Retinografia).
- **Diagnóstico Adaptativo:** Criação de sistemas de diagnóstico que podem evoluir e se adaptar a novas populações de pacientes ou novos protocolos de imagem sem a necessidade de retreinar o modelo do zero com os dados originais.

## Integration

O CLMS é um *framework* de pesquisa e, embora o código exato não tenha sido extraído, a implementação é baseada em técnicas de Deep Learning e requer:
1.  Um modelo pré-treinado no domínio de origem (Source Domain).
2.  Um conjunto de dados não rotulados do novo domínio de destino (Target Domain).
3.  Implementação do módulo de reconstrução multi-escala, da estratégia de aprendizagem contínua (para evitar o esquecimento) e do componente de alinhamento de estilo.
O artigo sugere que a implementação é feita em Python, utilizando bibliotecas padrão de Deep Learning (como PyTorch ou TensorFlow), e é um método de adaptação de domínio. O código-fonte geralmente é disponibilizado em repositórios do GitHub associados aos autores.

## URL

https://www.sciencedirect.com/science/article/pii/S1361841524003293