# MODIS (Moderate Resolution Imaging Spectroradiometer) - NASA Earth Observations

## Description
O Moderate Resolution Imaging Spectroradiometer (MODIS) é um instrumento científico fundamental a bordo dos satélites Terra (lançado em 1999) e Aqua (lançado em 2002) da NASA. Ele coleta dados continuamente em 36 bandas espectrais, abrangendo comprimentos de onda de 0,4 µm a 14,4 µm, com cobertura global a cada 1 a 2 dias. Os dados MODIS são essenciais para a compreensão dos processos ambientais e dinâmicos globais na Terra, fornecendo informações sobre a atmosfera, a terra e os oceanos. O instrumento ainda está operacional, e a NASA continua a migrar e fornecer acesso aos dados através da infraestrutura Earthdata Cloud.

## Statistics
**Versão Atual:** Collection 6.1 (V061). **Volume de Arquivo:** O LAADS DAAC, principal arquivo de dados MODIS, hospeda um volume de arquivo de mais de **20 Petabytes (PB)** de dados. **Satélites:** Terra (lançado em 1999) e Aqua (lançado em 2002). **Período de Cobertura:** Desde 2000 até o presente (instrumento ainda operacional). **Amostras/Granulometria:** Os dados são distribuídos em grânulos de 5 minutos (swath) ou em grades (tiled/CMG).

## Features
**Resolução Espacial:** 250 m, 500 m e 1000 m. **Resolução Espectral:** 36 bandas espectrais. **Resolução Temporal:** Cobertura global a cada 1-2 dias. **Produtos:** Mais de 40 produtos de dados de nível 1, 2 e 3, cobrindo a atmosfera (nuvens, aerossóis, vapor d'água), a terra (uso e cobertura do solo, temperatura da superfície, vegetação) e os oceanos (temperatura da superfície do mar, clorofila). **Disponibilidade:** Dados em tempo real e quase em tempo real (NRT) disponíveis em até 3 horas após a observação.

## Use Cases
O MODIS é amplamente utilizado em diversas disciplinas científicas, incluindo: **Monitoramento Climático:** Análise de nuvens, aerossóis e balanço de energia. **Saúde Vegetativa:** Estudos de saúde da vegetação, índice de área foliar (LAI) e produtividade primária. **Uso e Cobertura do Solo:** Mapeamento de mudanças na cobertura e uso do solo. **Oceanografia:** Medição da temperatura da superfície do mar e concentração de clorofila. **Monitoramento de Desastres:** Detecção e monitoramento de incêndios florestais, inundações e derramamentos de óleo em tempo real e quase em tempo real.

## Integration
O acesso aos dados MODIS é feito principalmente através do **LAADS DAAC (Level-1 and Atmosphere Archive and Distribution System)** da NASA. Os usuários podem descobrir e baixar dados usando a ferramenta de busca interativa do LAADS DAAC. Além disso, a NASA está ativamente migrando os dados MODIS para a infraestrutura **Earthdata Cloud** (hospedada na AWS), o que permite novas opções de acesso e processamento em nuvem, como o uso de tokens Bearer para acesso programático. O acesso programático é recomendado para grandes volumes de dados, utilizando scripts de download ou APIs. A versão de coleção mais recente é a **Collection 6.1 (V061)**.

## URL
[https://www.earthdata.nasa.gov/data/instruments/modis](https://www.earthdata.nasa.gov/data/instruments/modis)
