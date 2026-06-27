# Landsat (Imagens de Satélite)

## Description
O programa Landsat é uma série de missões de satélites de observação da Terra, gerenciadas conjuntamente pela NASA e pelo U.S. Geological Survey (USGS). Desde 1972, os satélites Landsat têm adquirido continuamente imagens da superfície terrestre, fornecendo um arquivo de dados ininterrupto. Este arquivo é considerado o "padrão ouro" para dados de satélite de média resolução (30 metros) devido à sua calibração rigorosa e documentada. Os dados são abertos e gratuitos desde 2008, o que impulsionou o desenvolvimento de novos produtos e aplicações, especialmente em Machine Learning e Inteligência Artificial. O programa é fundamental para monitorar mudanças na superfície terrestre, recursos naturais e o meio ambiente.

## Statistics
**Volume Diário:** Cerca de **1 terabyte (TB)** de novos dados de missão Landsat são adquiridos e baixados diariamente. Após o processamento, cerca de **3 TB** são adicionados ao arquivo USGS EROS por dia. **Versões:** Os dados estão organizados em **Coleções**, sendo a **Collection 2** a versão operacional atual (introduzida em dezembro de 2020), que inclui produtos de Nível 1 e Nível 2 (Reflectância e Temperatura de Superfície). **Amostras/Cenas:** O arquivo total é vasto, com o número de cenas de Nível 1 e Nível 2 da Collection 2 crescendo continuamente através do programa Global Archive Consolidation (LGAC). O volume total de dados baixados do arquivo USGS EROS desde dezembro de 2008 (FY 2009) até julho de 2025 (FY 2025) é medido em petabytes.

## Features
**Resolução Espacial:** Média (30 metros para a maioria das bandas). **Cobertura:** Global e contínua desde 1972 (série temporal longa). **Bandas Espectrais:** Varia de 7 a 11 bandas (dependendo do satélite, como Landsat 8 e 9), incluindo visível, infravermelho próximo (NIR) e infravermelho termal (TIRS). **Produtos:** Inclui produtos de Nível 1 (baseados em cena), Nível 2 (Reflectância de Superfície e Temperatura de Superfície) e Dados Prontos para Análise (ARD) dos EUA. **Acesso:** Livre e aberto. **Calibração:** Considerado o "padrão ouro" para dados de média resolução devido à calibração rigorosa.

## Use Cases
**Monitoramento Ambiental:** Mapeamento e monitoramento de mudanças na cobertura terrestre, desmatamento, expansão urbana, agricultura e saúde da vegetação. **Agricultura de Precisão:** Otimização de rendimento, uso de recursos e sustentabilidade através da análise de dados geoespaciais. **Recursos Hídricos:** Monitoramento de corpos d'água, extensão de águas superficiais e detecção de florações de algas. **Estudos Climáticos:** Análise de tendências de longo prazo em temperatura de superfície e mudanças glaciais. **Inteligência Artificial/Machine Learning:** Os dados Landsat são amplamente utilizados como dataset de treinamento e validação para modelos de Deep Learning em sensoriamento remoto, como detecção de mudanças, classificação de uso do solo e segmentação semântica.

## Integration
Os dados Landsat são acessíveis gratuitamente através de várias plataformas e ferramentas:
*   **USGS EarthExplorer (EE):** Interface gráfica para definir áreas de interesse, selecionar datas e pesquisar múltiplos datasets simultaneamente. Permite o download de produtos Nível 1, Nível 2 e ARD.
*   **Landsat in the Cloud (AWS):** Acesso aos produtos operacionais Landsat Collection 2 na plataforma Amazon Web Services (AWS), utilizando metadados SpatioTemporal Asset Catalog (STAC).
*   **USGS ESPA (EROS Science Processing Architecture):** Interface sob demanda para solicitar índices espectrais (NDVI, SAVI, etc.) e produtos de nível superior (Reflectância Aquática, Evapotranspiração Real). Suporta Bulk API e Downloader para recuperação de dados em massa.
*   **API M2M (Machine to Machine):** API RESTful JSON para acesso programático aos datasets, oferecendo as mesmas opções do EarthExplorer para scripts.
*   **Outras Ferramentas:** Acesso e visualização também são possíveis através do Sentinel Hub EO Browser e Esri Landsat Explorer.

## URL
[https://www.usgs.gov/landsat-missions/data](https://www.usgs.gov/landsat-missions/data)
