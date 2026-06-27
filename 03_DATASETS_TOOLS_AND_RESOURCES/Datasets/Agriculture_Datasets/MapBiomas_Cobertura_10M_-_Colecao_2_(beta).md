# MapBiomas Cobertura 10M - Coleção 2 (beta)

## Description

Mapas anuais de cobertura e uso da terra para o Brasil com resolução espacial de 10 metros, utilizando imagens do satélite Sentinel-2. A coleção abrange o período de 2016 a 2023. É uma versão BETA, mas oferece maior detalhe espacial em comparação com a coleção principal de 30m, sendo ideal para detalhamento de uso agropecuário e estimativa de biomassa em nível local.

## Statistics

Resolução espacial de 10 metros. Período de 2016 a 2023. Cobertura para todo o Brasil.

## Features

Alta resolução espacial (10m), ideal para detalhamento de uso agropecuário e florestas ripárias. Baseado em imagens Sentinel-2. Utiliza a mesma legenda da Coleção 9 (até o nível 3).

## Use Cases

Detalhamento de uso agropecuário, monitoramento de áreas de preservação permanente (APP), estudos de mudança de uso da terra em alta resolução, estimativa de biomassa em nível local.

## Integration

Acesso via Google Earth Engine (GEE) Asset: projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1. Download direto de GeoTiff por ano (2019 a 2023) via Google Cloud Storage. Exemplo de URL para 2023: https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/lulc_10m/collection_2/integration/mapbiomas_10m_collection2_integration_v1-classification_2023.tif.

Exemplo de código GEE (pseudo-código):
```python
import ee
ee.Initialize()
mapbiomas_10m = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1')
lulc_2023 = mapbiomas_10m.select('classification_2023')
# ... processamento e exportação
```

## URL

https://brasil.mapbiomas.org/mapbiomas-cobertura-10m/