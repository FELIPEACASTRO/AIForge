# World Ocean Database (WOD) - NOAA Ocean Data

## Description
O **World Ocean Database (WOD)** é a maior e mais abrangente coleção mundial de dados de perfil oceânico in situ, mantida pelo National Centers for Environmental Information (NCEI) da NOAA. O WOD é uma ferramenta essencial para pesquisa oceanográfica, climática e ambiental, fornecendo dados de alta qualidade, formatados uniformemente e controlados, que abrangem o período de 1772 até o presente. A versão mais recente, WOD23, foi lançada em 2024 e inclui mais de 20 milhões de "casts" oceanográficos. O dataset é fundamental para a análise climática oceânica de longo prazo e para o desenvolvimento de modelos de IA em ciências da Terra.

## Statistics
- **Versão Mais Recente:** WOD23 (World Ocean Database 2023), lançado em 2024.
- **Casts Oceanográficos:** Mais de 20.6 milhões de "casts" (medições de perfil).
- **Observações:** Cerca de 3.6 bilhões de observações individuais de perfil.
- **Período de Tempo:** Abrange de 1772 até o presente (WOD23 inclui dados até o final de 2022).
- **Atualizações:** Lançamentos principais periódicos e atualizações trimestrais.

## Features
- **Variáveis Abrangentes:** Inclui até 27 perfis de variáveis oceânicas essenciais, como temperatura, salinidade, oxigênio dissolvido, nutrientes (fosfato, nitrato, silicato), clorofila, pH e variáveis de estado do mar.
- **Controle de Qualidade:** Os dados são rigorosamente controlados por qualidade e são reproduzíveis a partir dos dados originais arquivados no NCEI.
- **Formato Interoperável:** Disponível em formatos interoperáveis, incluindo NetCDF (Climate and Forecast compliant ragged-array), ideal para processamento por modelos de Machine Learning e Deep Learning.
- **Metadados Granulares:** Possui metadados extensivos que facilitam a busca e o uso dos dados.

## Use Cases
- **Modelagem Climática:** Análise de longo prazo e histórica do clima oceânico, essencial para estudos de aquecimento global e acidificação dos oceanos.
- **Pesquisa Oceanográfica:** Estudo da circulação oceânica, da distribuição de propriedades da água e dos ecossistemas marinhos.
- **Inteligência Artificial:** Treinamento de modelos de Machine Learning e Deep Learning para previsão de condições oceânicas, detecção de anomalias e reconstrução de dados ausentes (como no projeto de reconstrução de oxigênio dissolvido usando ML).
- **Desenvolvimento de Produtos:** Criação de produtos de dados científicos e socioeconômicos de valor agregado, como o World Ocean Atlas (WOA).

## Integration
O acesso ao WOD é facilitado por diversas vias:
1.  **WODselect:** Um sistema de recuperação online que permite aos usuários buscar dados por parâmetros específicos (data, área geográfica, tipo de sonda) e variáveis medidas, com a opção de baixar datasets customizados nos formatos WOD nativo, CSV ou **NetCDF**.
2.  **Acesso Direto:** Dados pré-organizados por localização e tempo estão disponíveis para download direto.
3.  **Plataformas Cloud (NODD):** O dataset é disponibilizado através do NOAA Open Data Dissemination (NODD) em plataformas de Cloud Service Providers (CSPs) como Amazon Web Services (AWS) e Google Cloud, facilitando o acesso e processamento em larga escala para aplicações de IA.

## URL
[https://www.ncei.noaa.gov/products/world-ocean-database](https://www.ncei.noaa.gov/products/world-ocean-database)
