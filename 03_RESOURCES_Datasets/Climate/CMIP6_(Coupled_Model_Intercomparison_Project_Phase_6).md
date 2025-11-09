# CMIP6 (Coupled Model Intercomparison Project Phase 6)

## Description
O **Coupled Model Intercomparison Project Phase 6 (CMIP6)** é um projeto internacional do World Climate Research Programme (WCRP) que fornece um conjunto coordenado de simulações de modelos climáticos globais (GCMs) para entender as mudanças climáticas passadas, presentes e futuras. É a principal fonte de dados para as avaliações do Painel Intergovernamental sobre Mudanças Climáticas (IPCC), incluindo o Sexto Relatório de Avaliação (AR6). O CMIP6 introduziu um design experimental mais complexo, incluindo os experimentos **DECK** (Diagnosis, Evaluation, and Characterization of Klima) e um conjunto de **MIPs** (Model Intercomparison Projects) endossados, que exploram diferentes aspectos do sistema climático e cenários socioeconômicos compartilhados (SSPs). Os dados são essenciais para a comunidade científica global que estuda o clima e seus impactos.

## Statistics
- **Tamanho Total:** Aproximadamente **24.5 PB** (Petabytes) de dados.
- **Contagem de Datasets:** Mais de **6.4 milhões** de datasets individuais.
- **Versões/Modelos:** **322** experimentos de **132** modelos CMIP6 registrados.
- **Atualizações:** O projeto está em andamento, com dados sendo continuamente publicados e atualizados nos nós do ESGF. Versões recentes (2023-2025) de datasets derivados (e.g., downscaled) continuam a ser lançadas.

## Features
- **Ampla Cobertura:** Inclui 322 experimentos de 132 modelos climáticos globais (GCMs) registrados, provenientes de 48 instituições científicas em 26 países.
- **Estrutura de Dados:** Os dados de saída são armazenados em arquivos **netCDF**, com uma variável por arquivo, em conformidade com as convenções CF (Climate and Forecast) e padronizados por "vocabulários controlados" (CVs).
- **Cenários Futuros:** Utiliza os **Shared Socio-economic Pathways (SSPs)** para projetar o clima futuro sob diferentes cenários de emissões de gases de efeito estufa e desenvolvimento socioeconômico.
- **Resolução Aprimorada:** Modelos CMIP6 frequentemente apresentam resoluções espaciais e temporais mais altas em comparação com as fases anteriores (CMIP5).
- **Dados de Forçamento:** Inclui dados de forçamento padrão (e.g., concentrações de gases de efeito estufa, aerossóis) para garantir a comparabilidade entre os modelos.

## Use Cases
- **Avaliação de Impactos Climáticos:** Principalmente para fornecer projeções climáticas para o IPCC e estudos de impacto em escala global e regional.
- **Downscaling e Regionalização:** Criação de datasets de alta resolução (e.g., 1 km) para estudos regionais de variáveis como temperatura e precipitação.
- **Modelagem de Recursos Hídricos:** Previsão de vazão de rios e avaliação de secas sob cenários futuros (SSPs).
- **Aplicações em Machine Learning (ML):** Uso dos dados CMIP6 como entrada para modelos de ML (e.g., Random Forest, Deep Learning) para aprimorar a previsão de variáveis climáticas e seus impactos em setores como a agricultura.
- **Estudos de Extremos Climáticos:** Projeção de mudanças na frequência e intensidade de eventos climáticos extremos.
- **Análise de Sensibilidade Climática:** Investigação da resposta do sistema climático a diferentes forçantes (e.g., aerossóis, gases de efeito estufa).

## Integration
O acesso aos dados do CMIP6 é gerenciado pela **Earth System Grid Federation (ESGF)**, uma rede distribuída de nós de dados globais (e.g., LLNL/EUA, DKRZ/Alemanha, CEDA/Reino Unido, IPSL/França). O acesso é facilitado pela interface web **Metagrid** (substituindo o CoG).

**Métodos de Download:**
1.  **Wget Script:** Scripts de download em lote podem ser gerados diretamente na interface Metagrid.
2.  **Globus Transfer:** Método recomendado para downloads de grandes volumes de dados, oferecendo melhor desempenho. Requer uma conta ESGF e autenticação via Globus Auth (suporta Google, GitHub e contas institucionais).
3.  **API RESTful:** Usuários avançados podem utilizar a API RESTful de Pesquisa do ESGF para acesso programático.

**Requisito:** É necessário criar uma conta ESGF (utilizando Globus Auth) para downloads em lote e transferências Globus. Os usuários devem aderir aos Termos de Uso do CMIP6, que exigem citação e reconhecimento adequados dos dados.

## URL
[https://wcrp-cmip.org/](https://wcrp-cmip.org/)
