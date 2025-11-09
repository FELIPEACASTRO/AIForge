# Google Cloud Public Datasets (BigQuery)

## Description

Os **Google Cloud Public Datasets** são um vasto repositório de conjuntos de dados públicos hospedados no **BigQuery**, o data warehouse sem servidor, altamente escalável e de baixo custo do Google Cloud. O valor único reside em permitir que usuários e cientistas de dados consultem petabytes de dados complexos (como genômica, dados climáticos, financeiros e de tráfego) usando SQL padrão, sem a necessidade de configurar infraestrutura, fazer upload de dados ou pagar pelo armazenamento. O programa é mantido pelo Google, que cobre os custos de armazenamento, e oferece acesso gratuito ao processamento do primeiro 1 TB de dados por mês, tornando-o uma ferramenta inestimável para pesquisa, análise e prototipagem de Machine Learning.

## Statistics

**Custo:** O Google paga pelo armazenamento. O processamento de consultas é gratuito até 1 TB por mês por usuário. **Tamanho:** O repositório contém petabytes de dados. Alguns datasets individuais, como o Common Crawl, podem ter mais de 170 TB. **Acessibilidade:** Acesso imediato a centenas de datasets públicos de alta qualidade. **Localização:** Os datasets são armazenados em locais multirregionais, como `US` ou `EU`.

## Features

**Acesso via SQL:** Permite a consulta de petabytes de dados usando o SQL padrão (GoogleSQL ou Legacy SQL). **Infraestrutura Gerenciada:** Não requer configuração de infraestrutura, upload de dados ou gerenciamento de armazenamento. **Variedade de Dados:** Inclui dados de diversas áreas como ciência, finanças, saúde, clima e governo. **Integração Nativa:** Acesso direto via Google Cloud Console, ferramenta de linha de comando `bq` e APIs REST. **Modelo de Custo:** O Google paga pelo armazenamento; o usuário paga apenas pelas consultas (com 1 TB de processamento gratuito por mês).

## Use Cases

**Pesquisa Científica:** Análise de dados genômicos (como o NIH chest x-ray dataset) e dados climáticos (como o GSOD da NOAA) para descobertas científicas. **Análise Financeira:** Utilização de dados de mercado e transações para modelagem preditiva e backtesting. **Machine Learning:** Treinamento de modelos de ML em grande escala com datasets prontos, como o de revisão do Wikipedia ou o de tráfego de GitHub. **Educação e Prototipagem:** Permite que estudantes e desenvolvedores experimentem o BigQuery e o SQL em grandes volumes de dados sem custos iniciais de infraestrutura. **Análise de Mídia:** Consulta a dados de grandes volumes de texto, como o dataset de trigramas ou o de notícias.

## Integration

O acesso e a integração são realizados através da API BigQuery, suportada por diversas bibliotecas de cliente. O método mais comum é via Python, utilizando a biblioteca `google-cloud-bigquery`.

**Exemplo de Integração em Python:**
```python
# Instalação: pip install google-cloud-bigquery
from google.cloud import bigquery

# Inicializa o cliente BigQuery
client = bigquery.Client()

# Consulta SQL para o dataset de nomes dos EUA
query = """
    SELECT
        name,
        sum(number) AS total_births
    FROM
        `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE
        state = 'TX'
    GROUP BY
        name
    ORDER BY
        total_births DESC
    LIMIT 10
"""

query_job = client.query(query)  # Envia a requisição à API

print("Os 10 nomes mais populares no Texas (1910-2013):")
for row in query_job:
    print(f"Nome: {row['name']}, Nascimentos: {row['total_births']}")
```

**Acesso via Console:**
1.  Navegue até o BigQuery no Google Cloud Console.
2.  Clique em **+ ADD DATA** e selecione **Explore public datasets**.
3.  Pesquise e adicione o dataset desejado ao seu projeto.

**Acesso via Linha de Comando (`bq`):**
```bash
# Executa uma consulta e salva o resultado em um arquivo CSV
bq query --use_legacy_sql=false --format=csv "SELECT word, word_count FROM \`bigquery-public-data.samples.shakespeare\` WHERE word = 'love'" > love_count.csv
```

## URL

https://docs.cloud.google.com/bigquery/public-data