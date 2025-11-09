# Sentinel (ESA Satellite Data) - Programa Copernicus

## Description
O programa Sentinel é o componente de observação da Terra do programa Copernicus da União Europeia, gerenciado pela Agência Espacial Europeia (ESA). Consiste em uma constelação de satélites (Sentinel-1, -2, -3, -5P, etc.) que fornecem dados de observação da Terra de forma contínua e gratuita. Os dados abrangem uma ampla gama de aplicações, desde o monitoramento de terras e oceanos até a qualidade do ar e gestão de emergências. O foco principal é fornecer dados para os serviços operacionais do Copernicus, garantindo uma cobertura global e um alto tempo de revisita.

## Statistics
O arquivo total de dados do Copernicus Sentinel é massivo, com o Copernicus Data Space Ecosystem (CDSE) reportando um total de **20.8 Petabytes (PB)** de produtos disponíveis em janeiro de 2024. O volume de dados cresce a uma taxa de **Terabytes por dia**. Por exemplo, um grânulo de Nível-1B do Sentinel-2 tem aproximadamente 27 MB. O programa Sentinel-1 publica mais de 95.000 produtos mensalmente. **Versões:** O programa é contínuo, com lançamentos e substituições de satélites (ex: Sentinel-2C lançado em setembro de 2024, substituindo o Sentinel-2A em janeiro de 2025).

## Features
**Constelação de Missões:** Inclui missões de radar (Sentinel-1), ópticas multiespectrais (Sentinel-2), monitoramento de oceanos e terras (Sentinel-3), e monitoramento atmosférico (Sentinel-5P). **Alta Resolução e Revisit:** O Sentinel-2, por exemplo, oferece 13 bandas espectrais com resoluções de 10m, 20m e 60m, e um tempo de revisita de 5 dias no Equador com dois satélites. **Acesso Livre e Aberto:** Todos os dados são fornecidos de forma gratuita e aberta para usuários globais. **Níveis de Processamento:** Os produtos variam de Nível-1C (Reflectância no Topo da Atmosfera) a Nível-2A (Reflectância na Superfície Corrigida Atmosfericamente), sendo este último compatível com CEOS Analysis Ready Data (CEOS-ARD).

## Use Cases
**Monitoramento de Terras:** Mapeamento de uso e cobertura do solo, agricultura de precisão, e monitoramento de florestas (silvicultura). **Monitoramento Marinho:** Observação de oceanos, gelo marinho, poluição e cor da água. **Gestão de Emergências:** Mapeamento de inundações, incêndios florestais e desastres naturais. **Clima e Atmosfera:** Monitoramento de gases de efeito estufa e qualidade do ar (Sentinel-5P). **Segurança:** Aplicações de segurança e vigilância marítima.

## Integration
O acesso aos dados é feito principalmente através do **Copernicus Data Space Ecosystem (CDSE)**, que substituiu o Copernicus Open Access Hub. O CDSE oferece diversas opções de acesso: **Download Direto:** Através do portal web do CDSE. **APIs:** Utilização de APIs RESTful, como a Sentinel Hub API e as Streamlined Data Access APIs (SDA), para acesso programático e download de produtos. **Plataformas de Processamento:** Integração com plataformas de análise em nuvem como JupyterLab, openEO e Sentinel Hub, permitindo o processamento de dados sem a necessidade de download local de grandes volumes. **Software:** Ferramentas como o SNAP (Sentinel Application Platform) da ESA são recomendadas para o processamento e visualização dos dados.

## URL
[https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
