# Weather Station Data (ICBA)

## Description
O dataset "Weather Station Data" do International Center for Biosaline Agriculture (ICBA) é uma coleção de dados de temperatura do ar diária e horária, coletados em sua estação meteorológica. O conjunto de dados é atualizado anualmente e disponibilizado em arquivos mensais no formato XLSX. É uma fonte primária de dados climáticos de alta frequência para a região de Dubai, Emirados Árabes Unidos. O portal também oferece uma versão com dados horários.

## Statistics
**Versões:** Anual (2020, 2021, 2022, 2023, 2024, 2025 - parcial) e Horária. **Formato:** XLSX (planilhas mensais). **Tamanho/Amostras:** Não especificado, mas consiste em aproximadamente 365 registros diários por ano (ou 8760 registros horários por ano) para a temperatura do ar. A versão de 2024, por exemplo, contém 12 arquivos mensais.

## Features
Dados de temperatura do ar (diária e horária). A granularidade é diária para as versões anuais e horária para a versão específica. Os dados são estruturados em planilhas XLSX, com campos como data, hora (para a versão horária) e temperatura (°C).

## Use Cases
Modelagem e previsão do tempo e clima local. Análise de tendências climáticas e mudanças sazonais. Pesquisa em agricultura de sequeiro e resiliência de culturas em ambientes salinos. Treinamento de modelos de Machine Learning para previsão de temperatura. Estudos de eficiência energética e planejamento urbano.

## Integration
O dataset pode ser acessado e baixado diretamente do portal de dados abertos do ICBA. Os dados são organizados por ano e mês. Para uso em projetos de Machine Learning ou análise de dados, os arquivos XLSX devem ser baixados e processados (por exemplo, convertidos para CSV ou DataFrame) usando bibliotecas como Pandas no Python. O uso é regido pela licença Open Data Commons Attribution License.

## URL
[https://data.biosaline.org/dataset/?res_format=XLSX](https://data.biosaline.org/dataset/?res_format=XLSX)
