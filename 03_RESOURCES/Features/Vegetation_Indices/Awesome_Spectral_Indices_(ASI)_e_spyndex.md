# Awesome Spectral Indices (ASI) e spyndex

## Description

O "Awesome Spectral Indices" (ASI) é um catálogo padronizado e uma biblioteca Python (`spyndex`) que compila e gerencia uma vasta coleção de índices espectrais para aplicações de Sensoriamento Remoto, incluindo aqueles que utilizam as bandas Red Edge (RE), Infravermelho Próximo (NIR) e Infravermelho de Ondas Curtas (SWIR). Publicado em 2023, o ASI visa padronizar o uso de índices espectrais, fornecendo fórmulas, bandas requeridas e referências para cada índice. É uma ferramenta essencial para a engenharia de features em projetos de Biomassa e Agropecuária, facilitando a aplicação de índices como NDVI, EVI, CIRE (Chlorophyll Index Red Edge) e NDWI (que frequentemente usa SWIR).

## Statistics

Mais de 200 índices espectrais catalogados. Publicado em 2023. A biblioteca `spyndex` é compatível com Python e integra-se com plataformas como Google Earth Engine (via `eemont`). O catálogo é continuamente atualizado pela comunidade. As bandas espectrais padronizadas incluem 17 parâmetros, cobrindo RE, NIR e SWIR para Landsat, Sentinel-2 e MODIS.

## Features

Catálogo padronizado de mais de 200 índices espectrais. Inclui índices que utilizam as bandas Red Edge (RE1, RE2, RE3), NIR (N, N2) e SWIR (S1, S2) de sensores como Sentinel-2 e Landsat-8/9. Fornece metadados completos para cada índice: nome curto, nome longo, fórmula, bandas requeridas, plataformas compatíveis e domínio de aplicação (e.g., vegetação, água, solo). A biblioteca Python (`spyndex`) permite o cálculo eficiente dos índices em arrays de dados.

## Use Cases

**Estimativa de Biomassa e Carbono:** Uso de índices como o CIRE (Chlorophyll Index Red Edge) e índices baseados em SWIR para estimar o teor de clorofila e o teor de água, que são proxies diretos para a Biomassa Acima do Solo (AGB) em florestas e pastagens. **Monitoramento da Saúde da Planta:** Detecção precoce de estresse hídrico (usando índices SWIR) e deficiência de nutrientes (usando índices Red Edge). **Mapeamento de Culturas e Pastagens:** Classificação e discriminação de diferentes tipos de cobertura do solo e estágios de degradação de pastagens, aproveitando a sensibilidade do Red Edge. **Modelagem de Produtividade:** Integração de índices espectrais como features em modelos de Machine Learning para prever a Produtividade Primária Bruta (GPP) e o rendimento das culturas.

## Integration

A integração é feita através da biblioteca Python `spyndex`.

**Instalação:**
```bash
pip install spyndex
```

**Exemplo de Uso (Cálculo de NDVI e CIRE com bandas Sentinel-2):**
```python
import spyndex
import numpy as np

# Exemplo de dados de reflectância (valores simulados)
# N: NIR (B8), R: Red (B4), RE1: Red Edge 1 (B5)
reflectance_data = {
    "N": np.array([0.4, 0.3, 0.5]),
    "R": np.array([0.1, 0.05, 0.15]),
    "RE1": np.array([0.2, 0.15, 0.25])
}

# Calcular múltiplos índices de uma vez
indices = spyndex.compute(
    indices=["NDVI", "CIRE"],
    **reflectance_data
)

# Resultado (dicionário com os arrays de resultados)
# print(indices["NDVI"])
# print(indices["CIRE"])
```

O catálogo também é acessível em formatos JSON e CSV para integração em outras plataformas (e.g., Google Earth Engine, R).

## URL

https://github.com/awesome-spectral-indices/awesome-spectral-indices