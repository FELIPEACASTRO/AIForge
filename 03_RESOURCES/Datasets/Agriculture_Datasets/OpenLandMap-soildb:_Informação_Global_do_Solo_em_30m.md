# OpenLandMap-soildb: Informação Global do Solo em 30m

## Description

O OpenLandMap-soildb é um conjunto de dados global e dinâmico de informações do solo com resolução espacial de 30 metros, cobrindo o período de 2000 a 2022 e com projeções futuras. Ele utiliza Aprendizado de Máquina Espaço-Temporal (Quantile Regression Random Forest) treinado em uma vasta compilação de amostras de solo legadas (legacy soil samples) para fornecer previsões consistentes de variáveis-chave do solo. O foco principal é o mapeamento de propriedades do solo em profundidade (3D), com atualizações a cada 5 anos para variáveis dinâmicas como pH e Carbono Orgânico do Solo (SOC). Este recurso é fundamental para a agricultura de precisão e estudos de biomassa.

## Statistics

**Resolução Espacial:** 30 metros. **Período:** 2000–2022+. **Amostras de Treinamento:** 272.000 amostras para pH do solo em H2O; 363.000 amostras para textura (argila, silte, areia); 216.000 amostras para densidade de Carbono Orgânico do Solo (SOC). **Métricas de Validação (pH):** RMSE de 0.51 e CCC de 0.91. **Variáveis Importantes (pH):** Índice de Aridez CHELSA, precipitação anual e grau de salinidade. **Estimativa de Estoque de Carbono:** 461 Pg (Peta gramas) para 0–30 cm (2020–2022+).

## Features

**Variáveis Chave:** pH do solo em H2O, densidade aparente (bulk density), frações de textura do solo (argila, silte e areia), Carbono Orgânico do Solo (SOC) e tipos de solo (USDA soil taxonomy subgroups). **Metodologia:** Utiliza Machine Learning Espaço-Temporal (Quantile Regression Random Forest) para predições globais. **Resolução:** 30 metros de resolução espacial, compatível com dados de satélite de média e fina resolução (Landsat, Sentinel). **Cobertura:** Global, com mapeamento de propriedades do solo em intervalos de 5 anos para variáveis dinâmicas.

## Use Cases

**Agricultura de Precisão:** Mapeamento de zonas de manejo e otimização da aplicação de fertilizantes e corretivos (como calcário para correção de pH). **Modelagem de Biomassa e Carbono:** Previsão da biomassa vegetal e estimativa de estoques de Carbono Orgânico do Solo (SOC) para projetos de mitigação climática e sequestro de carbono. **Estudos Ambientais:** Monitoramento da degradação do solo e suporte a projetos de restauração de terras em escala global. **Modelagem Hidrológica:** Uso das frações de textura e densidade aparente para estimar a capacidade de retenção de água e a umidade do solo.

## Integration

Os produtos de dados resultantes podem ser acessados e baixados através do repositório Zenodo, com o DOI `10.5281/zenodo.15470431`. O conjunto de dados de treinamento também está disponível em `10.5281/zenodo.4748499`. Ambos são liberados sob licença CC-BY. A integração em projetos de IA e Biomassa é feita através do download dos *rasters* de 30m e sua incorporação em modelos de *machine learning* ou plataformas de GIS. O código-fonte e a metodologia estão disponíveis no GitHub do OpenLandMap.

## URL

https://essd.copernicus.org/preprints/essd-2025-336/