# BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

## Description

O BioMassters é um conjunto de dados de referência (benchmark) criado para investigar o potencial de séries temporais de satélite multimodais (Sentinel-1 SAR e Sentinel-2 MSI) na estimativa de Biomassa Acima do Solo (AGB) em larga escala. O dataset utiliza dados abertos de LiDAR aerotransportado do Centro Florestal Finlandês como referência de verdade terrestre. Foi lançado como parte de um desafio de aprendizado de máquina na plataforma DrivenData em 2023. O objetivo é promover o desenvolvimento de modelos de aprendizado profundo para produzir mapas de biomassa precisos e de alta resolução, superando as limitações de técnicas tradicionais de campo e LiDAR que são caras e difíceis de escalar.

## Statistics

Localização: Finlândia. Número de Amostras: Quase 13.000 patches (chips) de floresta. Tamanho do Patch: 2.560 x 2.560 metros, redimensionado para 256 x 256 pixels a 10 metros de resolução. Período de Coleta: 2016 a 2021. Referência Terrestre (Label): Biomassa Acima do Solo (AGB) por pixel, derivada de medições de LiDAR aerotransportado. Dados de Entrada: Séries temporais mensais de Sentinel-1 (SAR) e Sentinel-2 (MSI).

## Features

Dados multimodais de séries temporais: Combina dados de Radar de Abertura Sintética (SAR) do Sentinel-1 e imagens multiespectrais (MSI) do Sentinel-2. Resolução espacial de 10 metros. Cobertura temporal de 5 anos (2016 a 2021), com imagens mensais agregadas para um período de 12 meses (Setembro a Agosto) para cada chip. O alvo (label) é a estimativa de AGB por pixel (10x10m) dentro de cada chip.

## Use Cases

Modelagem de Biomassa: Treinamento e avaliação de modelos de aprendizado de máquina e aprendizado profundo para estimativa de AGB em florestas. Monitoramento de Carbono: Uso em inventários florestais e monitoramento da capacidade de sequestro de carbono. Pesquisa em Sensoriamento Remoto: Investigação do valor preditivo de dados multimodais (SAR e MSI) e séries temporais densas para aplicações ambientais. Competições de ML: Serve como um conjunto de dados de referência para desafios de ciência de dados e aprendizado de máquina.

## Integration

O conjunto de dados e o código associado estão disponíveis no site do projeto e no repositório GitHub. O acesso direto aos dados é feito através da Source Cooperative (mencionado na documentação do TorchGeo, que oferece uma interface para o dataset). O código dos competidores vencedores do desafio DrivenData está disponível no GitHub, servindo como exemplos de implementação e feature engineering.

**Exemplo de Acesso (via TorchGeo, indicando a fonte original):**
O dataset pode ser acessado programaticamente através de bibliotecas como TorchGeo, que aponta para a Source Cooperative como fonte de download.

**Estrutura de Arquivos:**
Os dados são fornecidos como GeoTIFFs, com o nome do arquivo seguindo o formato `{chip_id}_{satellite}_{month_number}.tif`.
- `S1`: Sentinel-1 (SAR)
- `S2`: Sentinel-2 (MSI)
- `month_number`: 00 (Setembro) a 11 (Agosto)

**Feature Engineering:**
As abordagens de feature engineering se concentram na extração de informações temporais e multimodais, como índices de vegetação (NDVI, EVI) a partir do Sentinel-2 e métricas de polarização e retroespalhamento (VV, VH) a partir do Sentinel-1, além de estatísticas temporais (médias, desvios, picos) ao longo do ano.

## URL

https://nascetti-a.github.io/BioMasster/