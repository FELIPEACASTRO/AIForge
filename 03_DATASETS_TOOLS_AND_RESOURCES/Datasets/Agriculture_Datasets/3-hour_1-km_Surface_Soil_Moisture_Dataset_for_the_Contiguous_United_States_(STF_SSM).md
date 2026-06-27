# 3-hour, 1-km Surface Soil Moisture Dataset for the Contiguous United States (STF_SSM)

## Description

Conjunto de dados de umidade do solo superficial (SSM) espacialmente contínuo (seamless) de 3 horas e 1 km para os Estados Unidos Contíguos (CONUS), desenvolvido usando um método de fusão espaço-temporal baseado em pares de imagens virtuais. O dataset, chamado STF_SSM, combina as vantagens do produto SMAP L4 SSM (alta resolução temporal de 3 horas, mas 9 km de resolução espacial) e do dataset Crop-CASMA (alta resolução espacial de 1 km, mas resolução temporal diária e com lacunas espaciais). O resultado é um dataset de longa série temporal (2015-2023) com alta resolução espacial e temporal, crucial para o monitoramento de secas e a validação de modelos de superfície terrestre. O dataset foi publicado em 2025, tornando-o um recurso de pesquisa muito recente.

## Statistics

- **Período de Cobertura:** 2015-2023.
- **Resolução:** 1 km (espacial) e 3 horas (temporal).
- **Validação:** Coeficiente de Correlação (CC) médio de 0.716 na escala diária e 0.689 na escala de 3 horas, em comparação com dados in-situ.
- **Erro:** Raiz do Erro Quadrático Médio não enviesado (ubRMSE) de 0.057 m³/m³ (diário) e 0.062 m³/m³ (3 horas).
- **Volume:** O dataset é composto por múltiplos arquivos de dados (Child Items) por ano, indicando um grande volume de dados geoespaciais.

## Features

- **Resolução Espacial:** 1 km (alta resolução).
- **Resolução Temporal:** 3 horas (alta resolução temporal, intra-diária).
- **Cobertura:** Estados Unidos Contíguos (CONUS).
- **Período:** 1º de abril de 2015 a 31 de dezembro de 2023.
- **Método:** Fusão espaço-temporal (Spatio-Temporal Fusion - STF) de dados SMAP L4 e Crop-CASMA.
- **Características:** Espacialmente contínuo (seamless), preenchendo lacunas de dados.

## Use Cases

- **Monitoramento de Secas:** Fornece insights sobre as rápidas mudanças na umidade do solo, essencial para a detecção e monitoramento de secas e períodos úmidos.
- **Modelagem de Superfície Terrestre:** Fonte de dados crítica para a calibração e validação de modelos hidrológicos e de superfície terrestre.
- **Agricultura de Precisão:** A alta resolução espacial (1 km) é valiosa para aplicações em agricultura de precisão, como otimização de irrigação e estimativa de produtividade.
- **Estudos de Ecossistemas:** Permite a análise das respostas dos ecossistemas às variações rápidas da umidade do solo.
- **Previsão do Tempo:** Aprimoramento da previsão do tempo e de eventos extremos (inundações).

## Integration

O dataset STF_SSM está disponível para download através do catálogo ScienceBase do USGS. Os dados são organizados em "Child Items" por ano. O acesso primário é via DOI e a página do ScienceBase.

**DOI:** `https://doi.org/10.5066/P13CCN69`

**Acesso via ScienceBase:**
1. Navegar para a página do ScienceBase.
2. Clicar nos links dos "Child Items" para baixar os arquivos de dados individuais (geralmente em formato NetCDF ou similar, comum para dados geoespaciais).
3. O código de geração do dataset (método STF) também está disponível para referência e reprodução em: `https://doi.org/10.6084/m9.figshare.28188011` (Yang et al., 2025b).

*Nota: O acesso direto aos arquivos de dados requer a navegação na página do ScienceBase para os links de download específicos de cada ano.*

## URL

https://www.sciencebase.gov/catalog/item/67ace224d34e329fb2046073