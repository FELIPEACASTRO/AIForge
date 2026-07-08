# Results - 2024 (TSE)

## Description
Official dataset from Brazil's Superior Electoral Court (Tribunal Superior Eleitoral, TSE) containing the detailed results of the 2024 Municipal Elections. It includes tabulation reports, vote-counting detail by municipality and zone/precinct, nominal and party-level vote counts, and votes by electoral precinct. It is the primary and most up-to-date source for the analysis of Brazilian electoral results.

## Statistics
The TSE does not disclose the total size in GB or the exact number of samples (rows) of the complete dataset. However, the official documentation warns that the CSV and TXT files contain a very large volume of data, exceeding Microsoft Excel's limit of 1,048,576 rows, indicating millions of records. The data is organized into multiple CSV files, separated by result type and by Federative Unit (UF).

## Features
Raw and detailed electoral results data. National coverage (Brazil) with granularity by electoral precinct, electoral zone, and municipality. Includes nominal votes, party votes, and tabulation reports. The data is made available in an open (CSV), non-proprietary format, and is updated daily during the election period.

## Use Cases
Analysis of voting patterns, studies of electoral behavior, academic research in political science and sociology, development of predictive and result-analysis models, data journalism, and monitoring of electoral integrity.

## Integration
The data can be downloaded directly from the TSE Open Data Portal in CSV format, with files separated by UF and result type. Due to the large volume, the use of statistical analysis tools (R, Python with Pandas), Business Intelligence (BI), or databases is recommended for processing. The TSE also provides an API for programmatic access to the data (DivulgaCandContas), although detailed documentation should be consulted for usage.

## URL
[https://dadosabertos.tse.jus.br/dataset/resultados-2024](https://dadosabertos.tse.jus.br/dataset/resultados-2024)
