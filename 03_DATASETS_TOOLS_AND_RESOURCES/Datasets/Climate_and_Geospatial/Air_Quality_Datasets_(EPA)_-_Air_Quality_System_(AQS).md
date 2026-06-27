# Air Quality Datasets (EPA) - Air Quality System (AQS)

## Description
O conjunto de dados de Qualidade do Ar da Agência de Proteção Ambiental dos EUA (EPA), conhecido como Air Quality System (AQS), é o repositório oficial de dados de qualidade do ar ambiente coletados em mais de 10.000 monitores nos Estados Unidos, Porto Rico e Ilhas Virgens Americanas. O AQS armazena medições de poluentes atmosféricos, incluindo os seis poluentes atmosféricos de critério (ozônio, material particulado (PM2.5 e PM10), monóxido de carbono, dióxido de enxofre, dióxido de nitrogênio e chumbo), além de dados de poluentes atmosféricos tóxicos e meteorológicos. Os dados são cruciais para a avaliação da qualidade do ar, modelagem de dispersão de poluentes e relatórios de conformidade com os Padrões Nacionais de Qualidade do Ar Ambiente (NAAQS).

## Statistics
**Monitores:** Mais de 10.000 monitores no total, com aproximadamente 5.000 ativos. **Séries Temporais:** Dados disponíveis desde 1990. **Amostras (Exemplo Anual):** O arquivo de concentração anual por monitor de 2023 contém 81.007 linhas de dados (4,132 KB compactado). O arquivo de descrição de monitores contém 367.437 linhas (6,701 KB compactado). **Atualizações:** Os arquivos pré-gerados são atualizados duas vezes por ano (junho e dezembro), com dados de 2025 já parcialmente disponíveis (até 31 de julho de 2025).

## Features
**Abrangência Geográfica:** Estados Unidos, Porto Rico e Ilhas Virgens Americanas. **Poluentes:** Inclui os seis poluentes de critério (O3, PM2.5, PM10, CO, SO2, NO2, Pb), além de poluentes atmosféricos tóxicos e dados meteorológicos. **Granularidade:** Dados disponíveis em diferentes níveis de agregação: amostras brutas, dados horários, diários e resumos anuais. **Formatos:** Arquivos CSV (comma separated variable) pré-gerados e compactados (.zip). **Metadados:** Inclui descrições detalhadas de locais (sites) e monitores.

## Use Cases
**Modelagem de IA:** Treinamento de modelos de Machine Learning (ML) e Deep Learning (DL) para previsão da qualidade do ar (ozônio, PM2.5, etc.), identificação de tendências e análise de séries temporais. **Pesquisa em Saúde Pública:** Estudo da correlação entre a poluição do ar e resultados de saúde, como doenças respiratórias e cardiovasculares. **Avaliação de Políticas:** Avaliação da eficácia das regulamentações ambientais e dos Padrões Nacionais de Qualidade do Ar Ambiente (NAAQS). **Modelagem de Exposição:** Criação de modelos de exposição à poluição do ar em alta resolução para comunidades específicas.

## Integration
O dataset é acessível principalmente através da página AirData da EPA, que oferece: 1. **Arquivos Pré-Gerados:** Arquivos ZIP contendo dados em formato CSV, agrupados por ano e tipo de agregação (anual, diário, horário). Os arquivos são atualizados duas vezes por ano (junho e dezembro). 2. **Ferramenta de Download Diário:** Uma ferramenta interativa para consultar e baixar dados diários para poluentes de critério. 3. **API:** Os dados também estão disponíveis via API para integração programática, embora a documentação da API deva ser consultada separadamente no site da EPA. Os usuários devem estar familiarizados com o programa de monitoramento da qualidade do ar da EPA para interpretar corretamente os dados.

## URL
[https://aqs.epa.gov/aqsweb/airdata/download_files.html](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
