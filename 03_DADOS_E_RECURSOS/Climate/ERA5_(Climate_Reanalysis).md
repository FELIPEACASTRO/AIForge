# ERA5 (Climate Reanalysis)

## Description
O ERA5 é a quinta geração de reanálise do Centro Europeu de Previsões Meteorológicas de Médio Prazo (ECMWF) para o clima e o tempo global. O dataset combina dados de modelos com observações de todo o mundo (assimilação de dados) para produzir um conjunto de dados globalmente completo e consistente, abrangendo as últimas oito décadas, com dados disponíveis a partir de 1940. Fornece estimativas horárias para um grande número de variáveis atmosféricas, de ondas oceânicas e da superfície terrestre. Inclui uma estimativa de incerteza amostrada por um conjunto subjacente de 10 membros.

## Statistics
**Cobertura Temporal:** 1940 até o presente (dados horários). **Resolução Horizontal (Reanálise):** 0.25° x 0.25° (atmosfera) e 0.5° x 0.5° (ondas oceânicas). **Resolução Temporal:** Horária. **Frequência de Atualização:** Diária (com latência de cerca de 5 dias para a versão preliminar ERA5T). **Formato de Arquivo:** GRIB (nativo), com versões otimizadas em Zarr disponíveis para ML/AI. **Versões:** ERA5 (principal), ERA5-Land (componente terrestre), ERA5T (preliminar), Anemoi training-ready version (otimizada para ML, 1979-2023).

## Features
O ERA5 oferece uma vasta gama de variáveis, incluindo componentes do vento (a 10m e 100m), temperatura, precipitação, radiação, e muitas outras quantidades da superfície terrestre, atmosféricas e de ondas oceânicas. O dataset principal tem uma resolução horizontal de 0.25° x 0.25° para a atmosfera e 0.5° x 0.5° para ondas oceânicas. Possui uma versão otimizada para Machine Learning (Anemoi) em formato Zarr.

## Use Cases
O ERA5 é amplamente utilizado em pesquisa e monitoramento climático, servindo como base para modelos de previsão do tempo e clima. É o dataset preferido para o treinamento de modelos de Machine Learning (ML) e Inteligência Artificial (IA) em previsão do tempo e downscaling espacial. Outros casos de uso incluem pesquisa climática retrospectiva, correção de viés em estimativas de evapotranspiração e estudos de impacto socioeconômico.

## Integration
O acesso primário é feito através do Copernicus Climate Data Store (CDS), utilizando a interface web ou programaticamente através do **CDS API service** (biblioteca Python `cdsapi`). Alternativamente, o dataset está disponível em plataformas como Google Earth Engine (agregados mensais, ERA5-Land) e AWS Open Data (em formatos GRIB e Zarr otimizado para nuvem). O uso do CDS API é o método recomendado para download em lote.

## URL
[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
