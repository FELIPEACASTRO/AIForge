# Top 100 Cryptocurrency (2020-2025)

## Description
Este é um dataset abrangente de preços diários de criptomoedas, cobrindo as 100 principais criptomoedas por capitalização de mercado. Ele fornece dados OHLC (Open, High, Low, Close) essenciais para análise de séries temporais e desenvolvimento de modelos preditivos. O conjunto de dados é valioso para pesquisadores e desenvolvedores interessados em finanças quantitativas e aprendizado de máquina aplicado ao mercado de ativos digitais. A inclusão da rede blockchain associada a cada ativo adiciona uma camada de informação útil para análises mais profundas.

## Statistics
**Tamanho do Dataset:** 11.9 MB (arquivo CSV único).
\n**Contagem de Amostras:** Mais de 200.000 linhas de dados.
\n**Versões:** Versão 1 (atualizada há 3 meses, a partir da data de pesquisa).
\n**Período de Cobertura:** 2020 a 2025.
\n**Granularidade:** Diária.
\n**Ativos:** 100 criptomoedas.

## Features
Dados de preço diário OHLC (Open, High, Low, Close); Abrange as 100 principais criptomoedas por capitalização de mercado; Inclui o campo "Blockchain Network" para cada ativo; Formato CSV de fácil utilização.

## Use Cases
Previsão de séries temporais e predição de preços de criptomoedas; Desenvolvimento de estratégias de negociação (trading) baseadas em IA/ML; Análise de correlação e impacto de sentimentos de notícias no mercado; Otimização de portfólio baseada em tendências históricas; Pesquisa acadêmica sobre a dinâmica do mercado de ativos digitais.

## Integration
O dataset está hospedado no Kaggle e pode ser baixado diretamente através do botão "Download" na página do recurso. Para usuários do Kaggle, a integração é facilitada pelo uso da API do Kaggle ou diretamente em Notebooks da plataforma. Para uso externo, o arquivo CSV deve ser baixado e importado para ambientes de desenvolvimento como Python (com Pandas) ou R.
\n\n**Exemplo de uso em Python (após download):**
\n```python
\nimport pandas as pd
\n
\n# Carregar o arquivo CSV
\ndf = pd.read_csv('top_100_cryptos_with_correct_network.csv')
\n
\n# Visualizar as primeiras linhas
\nprint(df.head())
\n
\n# Converter a coluna 'date' para o formato datetime
\ndf['date'] = pd.to_datetime(df['date'])
\n
\n# Análise básica (ex: preço médio de fechamento)
\nprint(df['close'].mean())
\n```

## URL
[https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025](https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025)
