# IMF Data (International Monetary Fund)

## Description
O **IMF Data** (Fundo Monetário Internacional) é uma vasta coleção de dados macroeconômicos e financeiros globais. O principal conjunto de dados, **International Financial Statistics (IFS)**, foi reestruturado e seus indicadores foram reorganizados em mais de 20 datasets temáticos menores (como CPI, BOP, EER, MFS, etc.) no Portal de Dados do FMI. Esses dados são cruciais para a análise da economia global, balanço de pagamentos, finanças governamentais e estatísticas monetárias. A reestruturação visa fornecer um modelo de dados mais consistente e granular.

## Statistics
**Cobertura:** Dados macroeconômicos e financeiros para mais de 180 países.
**Frequência:** Varia de Anual, Trimestral, Mensal a Diária, dependendo do dataset temático.
**Versões:** Os dados são atualizados continuamente. A estrutura de dados IFS foi reestruturada em mais de 20 datasets temáticos, com a notícia de acesso mais recente datada de Março de 2025. Não há um tamanho total de arquivo único, pois é um portal de dados dinâmico.

## Features
O IMF Data é composto por mais de 20 datasets temáticos que cobrem indicadores de mais de 180 países. Os principais tópicos incluem: Estatísticas da Força de Trabalho (LS), Índice de Preços ao Consumidor (CPI), Balanço de Pagamentos (BOP), Posição de Investimento Internacional (IIP), Estatísticas Monetárias e Financeiras (MFS), e Contas Econômicas Nacionais (NEA). Os dados são acessíveis via portal web e API, com suporte para formatos SDMX 2.1 e 3.0.

## Use Cases
Modelagem e previsão macroeconômica. Análise de estabilidade financeira e balanço de pagamentos. Pesquisa acadêmica e política sobre indicadores econômicos globais (PIB, inflação, taxas de câmbio, etc.). Desenvolvimento de modelos de IA/ML para prever tendências econômicas e financeiras e analisar o risco soberano.

## Integration
O acesso aos dados do FMI pode ser feito de várias maneiras:
1.  **Portal de Dados:** Acesso interativo e download via interface web em [https://data.imf.org/](https://data.imf.org/).
2.  **API (SDMX 2.1 e 3.0):** Acesso programático via API SDMX. O FMI recomenda a biblioteca Python `sdmx1` para consultas. Para acessar dados que faziam parte do IFS original, deve-se usar o filtro `c[IFS_Flag]=True` na consulta da API.
3.  **Outras Ferramentas:** MATLAB, R, Stata e Excel Add-In.
**Exemplo de Código Python (Acesso Público):**
```python
import sdmx
IMF_DATA = sdmx.Client('IMF_DATA')
# Exemplo: Acessar o dataset CPI para EUA e Canadá a partir de 2018
data_msg = IMF_DATA.data('CPI', key='USA+CAN.CPI.CP01.IX.M', params={'startPeriod': 2018})
cpi_df = sdmx.to_pandas(data_msg)
```

## URL
[https://data.imf.org/](https://data.imf.org/)
