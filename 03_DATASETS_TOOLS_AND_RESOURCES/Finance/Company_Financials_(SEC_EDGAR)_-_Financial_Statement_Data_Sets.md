# Company Financials (SEC EDGAR) - Financial Statement Data Sets

## Description
O conjunto de dados "Financial Statement Data Sets" da SEC (Securities and Exchange Commission) dos EUA fornece informações numéricas extraídas das demonstrações financeiras de todas as empresas que arquivam relatórios corporativos usando a linguagem XBRL (eXtensible Business Reporting Language) através do sistema EDGAR (Electronic Data Gathering, Analysis, and Retrieval). O dataset apresenta os dados em um formato "achatado" (flattened) para facilitar a análise e comparação de informações financeiras ao longo do tempo e entre diferentes registrantes. Ele inclui dados de balanços, demonstrações de resultados e fluxos de caixa.

## Statistics
O dataset é atualizado trimestralmente e abrange dados desde janeiro de 2009 até o trimestre mais recente (Junho de 2025, conforme a fonte). Cada arquivo trimestral tem um tamanho médio de aproximadamente 100 MB (variando de 75 MB a 122 MB nos trimestres recentes de 2023-2025). O número total de amostras (registros) é massivo, com milhões de entradas por trimestre, dependendo do arquivo (submissões, dados numéricos, etc.). A versão mais recente é a do 2º trimestre de 2025 (2025 Q2).

## Features
Dados numéricos de demonstrações financeiras (balanço, resultado, fluxo de caixa) de todas as empresas listadas nos EUA. Os dados são extraídos de relatórios XBRL (eXtensible Business Reporting Language) arquivados no sistema EDGAR. O formato é "achatado" (flattened) para facilitar a análise e comparação. Inclui campos adicionais como a Classificação Industrial Padrão (SIC) da empresa. O dataset é composto por quatro arquivos principais: `sub` (informações de submissão), `num` (dados numéricos), `tag` (definições de tags XBRL) e `pre` (estrutura de apresentação).

## Use Cases
Modelagem preditiva de falência ou desempenho financeiro. Análise de sentimento e risco de mercado com base em divulgações financeiras. Treinamento de Large Language Models (LLMs) para compreensão de documentos financeiros. Pesquisa acadêmica em finanças corporativas e contabilidade. Desenvolvimento de ferramentas de análise de dados financeiros e conformidade regulatória.

## Integration
Os dados são disponibilizados em arquivos ZIP trimestrais contendo arquivos de texto delimitados por tabulação (TXT). Para uso, é necessário baixar os arquivos ZIP do trimestre desejado e descompactá-los. A SEC também fornece uma API (Application Programming Interface) para acesso programático aos dados, permitindo que desenvolvedores consultem submissões EDGAR por empresa e dados XBRL extraídos. Existem bibliotecas de terceiros em linguagens como Python e R (por exemplo, `edgar` no R) que facilitam o download e a análise dos dados.

## URL
[https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets)
