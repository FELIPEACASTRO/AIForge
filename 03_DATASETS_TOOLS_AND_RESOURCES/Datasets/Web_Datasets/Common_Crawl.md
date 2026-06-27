# Common Crawl

## Description
Common Crawl é uma organização sem fins lucrativos (501(c)(3)) fundada em 2007 que mantém um repositório aberto e gratuito de dados de rastreamento da web. Seu objetivo é tornar a extração, transformação e análise em larga escala de dados abertos da web acessível a pesquisadores. O corpus total tem mais de 300 bilhões de páginas e é citado em mais de 10.000 artigos de pesquisa. É a maior fonte de dados de pré-treinamento para a maioria dos Grandes Modelos de Linguagem (LLMs) modernos.

## Statistics
Frequência: Mensal. Versão Mais Recente (Outubro 2025): CC-MAIN-2025-43. Tamanho (Outubro 2025): 2.61 bilhões de páginas web ou 468 TiB de conteúdo não compactado. Tamanho Total (Corpus): Mais de 300 bilhões de páginas, abrangendo 18 anos.

## Features
O dataset é disponibilizado em formatos brutos e processados, incluindo arquivos WARC (Web ARChive), WAT (Web Archive Transformation) e WET (Web Extracted Text). Inclui dados brutos da web, metadados e texto extraído. A organização também lança regularmente Web Graphs (nível de host e domínio) e anotações de qualidade como GneissWeb para filtragem de conteúdo.

## Use Cases
Treinamento de Grandes Modelos de Linguagem (LLMs) como GPT e LLaMA. Pesquisa acadêmica em mineração de dados da web, análise de tendências, estudos de preservação digital e linguística computacional. Análise de tendências e monitoramento da evolução da web.

## Integration
O acesso aos dados é gratuito e pode ser feito de duas formas principais: 1. **Amazon S3:** Os dados estão hospedados no bucket `s3://commoncrawl/` na região US-East-1 (Northern Virginia) da AWS. Para acesso via S3 API, é necessária autenticação. 2. **Download HTTP(S):** Para baixar arquivos diretamente sem uma conta AWS, use o prefixo `https://data.commoncrawl.org/` seguido do caminho do arquivo. Ferramentas como `wget` ou `curl` podem ser usadas. O Common Crawl também fornece ferramentas e bibliotecas para processamento dos dados.

## URL
[https://commoncrawl.org/](https://commoncrawl.org/)
