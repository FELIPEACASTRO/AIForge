# Fatores Topográficos (Elevação, Inclinação, Aspecto) como Features

## Description

Características topográficas derivadas de Modelos Digitais de Elevação (MDEs), como Elevação, Inclinação (Slope) e Aspecto, são utilizadas como features ambientais estáticas para aprimorar a precisão de modelos de Machine Learning (e.g., Random Forest) na estimativa de Biomassa Acima do Solo (AGB) e na previsão de rendimento de culturas. Esses fatores ajudam a explicar a heterogeneidade espacial e as condições microclimáticas que afetam o crescimento da vegetação e a distribuição de nutrientes/água.

## Statistics

A inclusão de fatores topográficos (inclinação e aspecto) em modelos de estimativa de biomassa de seringueira resultou em um aumento médio de 0.007 no R², uma redução média absoluta de 4.54 t/ha no Bias e uma redução média de 9.73% no RMSE. A Elevação e o MDE SRTM de 30m são frequentemente citados como fontes de dados.

## Features

Elevação (Altitude), Inclinação (Grau de declive), Aspecto (Orientação da encosta). São features estáticas que influenciam o microclima, a erosão do solo, a distribuição de água e nutrientes, e a incidência de radiação solar.

## Use Cases

Estimativa de Biomassa Acima do Solo (AGB) em florestas e plantações (e.g., seringueira, florestas de montanha). Previsão de rendimento de culturas (e.g., cana-de-açúcar, milho, trigo). Mapeamento de propriedades do solo e monitoramento de abandono de terras agrícolas. Modelagem de perda de solo por erosão hídrica (RUSLE).

## Integration

Os dados topográficos são tipicamente derivados de MDEs (como SRTM, ASTER GDEM, ou MDEs de alta resolução de UAV/LiDAR) usando ferramentas GIS (e.g., ArcGIS, QGIS, Google Earth Engine - GEE). O GEE é citado como fonte para o MDE SRTM de 30m. As features são então integradas como variáveis de entrada em modelos de Machine Learning (e.g., Random Forest, XGBoost, Deep Learning) juntamente com dados de sensoriamento remoto (e.g., índices de vegetação, dados multiespectrais).

## URL

https://www.sciencedirect.com/science/article/pii/S2666719325001955