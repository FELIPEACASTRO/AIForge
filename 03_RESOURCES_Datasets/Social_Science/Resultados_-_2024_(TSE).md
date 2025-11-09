# Resultados - 2024 (TSE)

## Description
Conjunto de dados oficial do Tribunal Superior Eleitoral (TSE) do Brasil contendo os resultados detalhados das Eleições Municipais de 2024. Inclui relatórios de totalização, detalhe da apuração por município e zona/seção, votação nominal e por partido, e votação por seção eleitoral. É a fonte primária e mais atualizada para análise dos resultados eleitorais brasileiros.

## Statistics
O TSE não divulga o tamanho total em GB ou o número exato de amostras (linhas) do conjunto completo. No entanto, a documentação oficial alerta que os arquivos CSV e TXT possuem um volume de dados muito grande, superando o limite de 1.048.576 linhas do Microsoft Excel, indicando milhões de registros. Os dados são organizados em múltiplos arquivos CSV, separados por tipo de resultado e por Unidade da Federação (UF).

## Features
Dados brutos e detalhados dos resultados eleitorais. Cobertura nacional (Brasil) com granularidade por seção eleitoral, zona eleitoral e município. Inclui votação nominal, votação por partido e relatórios de totalização. Os dados são disponibilizados em formato aberto (CSV), não proprietário, e são atualizados diariamente durante o período eleitoral.

## Use Cases
Análise de padrões de votação, estudos de comportamento eleitoral, pesquisa acadêmica em ciência política e sociologia, desenvolvimento de modelos preditivos e de análise de resultados, jornalismo de dados e monitoramento da integridade eleitoral.

## Integration
Os dados podem ser baixados diretamente do Portal de Dados Abertos do TSE em formato CSV, com arquivos separados por UF e tipo de resultado. Devido ao grande volume, é recomendado o uso de ferramentas de análise estatística (R, Python com Pandas), Business Intelligence (BI) ou bancos de dados para processamento. O TSE também disponibiliza uma API para acesso programático aos dados (DivulgaCandContas), embora a documentação detalhada deva ser consultada para uso.

## URL
[https://dadosabertos.tse.jus.br/dataset/resultados-2024](https://dadosabertos.tse.jus.br/dataset/resultados-2024)
