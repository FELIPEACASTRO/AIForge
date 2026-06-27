# DAVIS (Densely Annotated VIdeo Segmentation)

## Description
O DAVIS (Densely Annotated VIdeo Segmentation) é um conjunto de dados e *benchmark* fundamental para o problema de **Segmentação de Objetos em Vídeo (Video Object Segmentation - VOS)**. A versão mais utilizada, DAVIS 2017, expandiu a versão original (DAVIS 2016) para incluir anotações de múltiplos objetos por sequência. O dataset é crucial para o desenvolvimento e avaliação de algoritmos de VOS, tanto em cenários semi-supervisionados (onde a máscara do objeto é fornecida no primeiro *frame*) quanto não-supervisionados (onde nenhum *input* humano é fornecido). O dataset é conhecido pela alta qualidade de suas anotações e pela diversidade de cenários de vídeo.

## Statistics
- **DAVIS 2017:** 150 sequências de vídeo, totalizando 10.459 *frames* anotados e 376 instâncias de objetos.
- **DAVIS 2016:** 50 sequências de vídeo, totalizando 3.455 *frames* anotados.
- **Tamanho do Dataset (DAVIS 2017):** Aproximadamente 792.26 MiB (versão TFDS 480p) ou vários GBs para a versão Full-Resolution.
- **Resolução:** Anotações oficiais em 480p, mas imagens em resolução total (até 4K) disponíveis.
- **Versões:** DAVIS 2016, DAVIS 2017, e extensões como DAVIS-17 Moving. O desafio mais recente foi em 2020.

## Features
- **Análise Densa:** Anotações de segmentação de alta qualidade em nível de pixel para cada *frame* relevante.
- **Múltiplas Versões:** Inclui DAVIS 2016 (objeto único) e DAVIS 2017 (múltiplos objetos).
- **Resoluções Variadas:** Disponível em 480p para avaliação padrão e em resolução total (Full-Resolution, até 4K) para pesquisa.
- **Modos de Avaliação:** Suporta avaliação para VOS semi-supervisionado (com máscara do primeiro *frame*) e não-supervisionado (sem *input* humano).
- **Desafios Anuais:** Foi a base para desafios anuais (DAVIS Challenge) de 2017 a 2020, impulsionando o estado da arte.

## Use Cases
- **Segmentação de Objetos em Vídeo (VOS):** Principal caso de uso para treinar e avaliar modelos de VOS.
- **Rastreamento de Múltiplos Objetos (MOT):** O DAVIS 2017, com anotações de múltiplos objetos, é relevante para tarefas de rastreamento.
- **Visão Computacional em Tempo Real:** Desenvolvimento de algoritmos eficientes que operam em sequências de vídeo.
- **Pesquisa em Visão Computacional:** Utilizado como *benchmark* padrão para o estado da arte em segmentação de vídeo.
- **Aplicações:** Robótica, veículos autônomos, vigilância e edição de vídeo avançada.

## Integration
O dataset e o código de avaliação estão disponíveis no site oficial. O uso geralmente envolve:
1.  **Download:** Baixar os arquivos `TrainVal` e `Test-Dev/Test-Challenge` (imagens e anotações) do site oficial.
2.  **Código de Avaliação:** Utilizar os repositórios Python ou MATLAB fornecidos para carregar o dataset e avaliar os resultados do modelo.
3.  **Estrutura:** O dataset é organizado em sequências de vídeo, com subpastas para imagens (*JPEGImages*) e anotações (*Annotations*).
4.  **Uso em Frameworks:** O dataset também está disponível em formatos prontos para uso em bibliotecas como TensorFlow Datasets (TFDS).
**Link de Download:** O link direto para a seção de downloads do DAVIS 2017 é `https://davischallenge.org/davis2017/code.html`. É necessário aceitar os termos de uso.

## URL
[https://www.davischallenge.org/](https://www.davischallenge.org/)
