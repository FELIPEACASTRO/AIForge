# US Census Data (Dados do Censo dos EUA)

## Description
O US Census Data (Dados do Censo dos EUA) é um vasto conjunto de dados demográficos, sociais, econômicos e habitacionais coletados e divulgados pelo U.S. Census Bureau. Não se trata de um único dataset, mas de uma coleção de programas de pesquisa, como o Censo Decenal, a American Community Survey (ACS), e as Estimativas Populacionais. É a fonte primária de informações estatísticas sobre a população e a economia dos Estados Unidos, fornecendo dados essenciais para a tomada de decisões governamentais, planejamento urbano, pesquisa acadêmica e desenvolvimento de modelos de IA. Os dados são desagregados em vários níveis geográficos, desde o nacional até blocos censitários.

## Statistics
O volume de dados é imenso e varia por programa. A American Community Survey (ACS), uma das principais fontes de dados contínuos, possui uma amostra anual de cerca de **3,5 milhões de endereços**. As versões são contínuas, com lançamentos anuais (como os dados 2019-2023 da ACS de 5 anos, lançados em 2025) e decenais (Censo 2020). Os dados são organizados em tabelas e arquivos de microdados de uso público (PUMS). Não há um tamanho único de arquivo, mas a coleção total é medida em terabytes.

## Features
**Cobertura Abrangente:** Inclui dados demográficos (idade, sexo, raça, origem hispânica), econômicos (renda, emprego, pobreza), sociais (educação, migração, idioma) e habitacionais. **Granularidade Geográfica:** Dados disponíveis em diversos níveis, como nacional, estadual, condados, municipal e blocos censitários. **Atualização Contínua:** Programas como a American Community Survey (ACS) fornecem estimativas anuais, complementando o Censo Decenal. **Acesso via API:** O Census Data API permite consultas personalizadas e a integração de estatísticas em aplicações web ou móveis.

## Use Cases
**Modelagem Preditiva:** Criação de modelos de Machine Learning para prever tendências demográficas, econômicas e sociais. **Análise de Mercado:** Identificação de mercados-alvo, análise de poder de compra e distribuição de renda para planejamento de negócios. **Planejamento Urbano e Políticas Públicas:** Uso de dados geográficos detalhados para alocação de recursos, planejamento de infraestrutura e avaliação de políticas sociais. **Pesquisa Acadêmica:** Estudos em sociologia, economia, demografia e ciência política. **Treinamento de Modelos de IA:** Utilização de dados censitários para treinar modelos de IA que necessitam de uma base de dados representativa da população e suas características.

## Integration
O acesso primário aos dados é feito através do portal **data.census.gov** ou diretamente via **Census Data API**. Para desenvolvedores e pesquisadores, o U.S. Census Bureau oferece uma API que permite a criação de consultas personalizadas. É necessário obter uma chave de API (API Key) no site do Census Bureau para acessar os dados programaticamente. A API suporta a recuperação de dados estatísticos brutos, e pode ser combinada com os serviços TIGERweb (para limites geográficos) e Geocoder (para tradução de endereços em coordenadas). Bibliotecas de terceiros em Python (como `census` ou `cenpy`) facilitam a integração.

## URL
[https://www.census.gov/data.html](https://www.census.gov/data.html)
