# Biodiversity Datasets (GBIF)

## Description
O Global Biodiversity Information Facility (GBIF) é uma rede internacional e uma infraestrutura de dados financiada por governos mundiais, com o objetivo de fornecer acesso livre e aberto a dados de ocorrência de biodiversidade. Ele agrega dados de milhares de instituições em todo o mundo, tornando-os disponíveis para pesquisa, política e tomada de decisões. O GBIF atua como um hub de dados primários de biodiversidade, incluindo registros de ocorrência, listas de verificação e dados de eventos de amostragem, provenientes de coleções de museus, ciência cidadã e outras fontes [1].

## Statistics
**Registros de Ocorrência:** Mais de **3.1 bilhões** de registros de ocorrência de espécies (dado de Setembro de 2025) [3].
**Datasets:** Mais de **118.713** datasets publicados [1].
**Instituições Publicadoras:** Mais de **2.588** instituições publicadoras [1].
**Uso Científico (2023):** Mais de **1.700** artigos científicos publicados em 2023 utilizaram dados mediados pelo GBIF [4].
**Tendência (2024):** O total de registros de ocorrência ultrapassou **3 bilhões** em 2024 [5].

## Features
**Dados Primários de Biodiversidade:** Inclui registros de ocorrência (onde e quando uma espécie foi observada), listas de verificação taxonômicas e dados de eventos de amostragem.
**Acesso Aberto (Open Access):** Todos os dados são disponibilizados sob licenças abertas, facilitando o uso irrestrito para pesquisa e política.
**Padronização:** Os dados são padronizados para o formato **Darwin Core Archive (DwC-A)**, garantindo interoperabilidade.
**API Robusta:** Oferece uma API completa para acesso programático aos dados, permitindo a integração com ferramentas e sistemas externos [2].
**Monitoramento de Uso:** Possui um sistema de rastreamento de literatura que identificou mais de **10.000 usos** em artigos revisados por pares até Julho de 2024 [1].

## Use Cases
**Pesquisa Ecológica e Biológica:** Estudo da distribuição de espécies, modelagem de nicho ecológico, análise de padrões de biodiversidade em escalas taxonômicas, temporais e espaciais [1].
**Conservação e Política:** Monitoramento do estado da biodiversidade e progresso em direção a metas internacionais, como as da Convenção sobre Diversidade Biológica (CBD). O GBIF é uma fonte de dados chave para indicadores como o Índice de Status de Espécies [1].
**Mudanças Climáticas:** Avaliação dos impactos das mudanças climáticas na distribuição geográfica das espécies.
**Saúde Humana e Agricultura:** Aplicações relacionadas à distribuição de vetores de doenças e espécies de interesse agrícola [1].

## Integration
O acesso aos dados do GBIF pode ser feito de duas maneiras principais [2]:
1.  **GBIF.org (Download Direto):** Através da interface web, os usuários podem pesquisar, filtrar e baixar dados em três formatos principais: **Simple CSV** (seleção de termos comuns), **Darwin Core Archive (DwC-A)** (inclui dados originais e interpretação do GBIF) e **Species list** (lista distinta de nomes).
2.  **Acesso Programático (API):** O GBIF oferece uma API RESTful completa para downloads de ocorrência e acesso a metadados. Bibliotecas como **`rgbif`** (para R) e **`pygbif`** (para Python) facilitam a integração e o download de grandes volumes de dados diretamente em ambientes de análise. O GBIF também lançou uma API experimental de **SQL Downloads** para consultas mais complexas [2].
*   **Termos de Uso:** Os usuários devem concordar com o **Acordo de Usuário de Dados** do GBIF, que exige a citação correta dos dados utilizados [2].

## URL
[https://www.gbif.org/](https://www.gbif.org/)
