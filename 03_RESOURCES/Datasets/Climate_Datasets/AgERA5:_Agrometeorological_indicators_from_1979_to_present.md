# AgERA5: Agrometeorological indicators from 1979 to present

## Description

O AgERA5 é um conjunto de dados meteorológicos de superfície diários, derivado do reanálise ERA5 do ECMWF, especificamente adaptado para aplicações em agricultura e estudos agroecológicos. Ele fornece variáveis agrometeorológicas essenciais, como temperatura, precipitação e umidade, em uma resolução espacial de 0.1° x 0.1°, com cobertura global. O conjunto de dados é corrigido para uma topografia mais fina, tornando-o mais preciso para modelagem agrícola.

## Statistics

**Cobertura Temporal:** De 1979 até o presente (atualização diária).
**Resolução Temporal:** Diária.
**Resolução Espacial:** 0.1° x 0.1° (aproximadamente 10 km).
**Cobertura Geográfica:** Global.
**Tamanho:** Variável, dependendo da seleção de tempo e área. Contém 19 variáveis agrometeorológicas.

## Features

Dados meteorológicos diários de superfície (velocidade do vento a 10m, temperatura e umidade do ponto de orvalho a 2m, umidade relativa a 2m, temperatura a 2m, precipitação, radiação solar, etc.). Resolução espacial de 0.1° x 0.1° (aproximadamente 10 km). Correção para topografia fina. Disponível em formato NetCDF-4.

## Use Cases

**Modelagem de Culturas:** Fornece dados de entrada essenciais para modelos de crescimento e rendimento de culturas (ex: DSSAT, APSIM).
**Monitoramento de Secas:** Utilizado para calcular índices de seca e monitorar o estresse hídrico na agricultura.
**Avaliação de Impacto Climático:** Análise de como as mudanças climáticas afetam a produtividade agrícola.
**Previsão Agrometeorológica:** Base para serviços de aconselhamento agrícola e previsões de curto e longo prazo.

## Integration

O AgERA5 está disponível através do Climate Data Store (CDS) do Copernicus. O acesso é feito por meio de um formulário de download interativo na web ou programaticamente usando a API do CDS (Python). Bibliotecas como `ag5Tools` (R) também facilitam o download e a extração.

Exemplo de acesso via API CDS (Python):
```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'sis-agrometeorological-indicators',
    {
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'total_precipitation',
        ],
        'year': '2024',
        'month': '01',
        'day': [
            '01', '02', '03',
        ],
        'time_aggregation': 'daily_mean',
        'area': [
            -10, -50, -30, -30, # [N, W, S, E]
        ],
    },
    'download.nc')
```

## URL

https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview