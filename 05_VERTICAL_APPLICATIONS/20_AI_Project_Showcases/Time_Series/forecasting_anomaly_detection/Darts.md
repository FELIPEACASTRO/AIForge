# Darts

## Description

Darts é uma biblioteca Python de código aberto para previsão e detecção de anomalias em séries temporais, projetada para ser amigável e unificada. Ela oferece uma ampla gama de modelos, desde clássicos como ARIMA até redes neurais profundas (N-BEATS), todos com uma interface consistente de `fit()` e `predict()` similar ao scikit-learn. Sua proposta de valor única reside na facilidade de uso, suporte nativo a séries temporais multivariadas e a capacidade de treinar modelos globais em múltiplos conjuntos de dados.

## Statistics

Suporta mais de 20 modelos de previsão (estatísticos e de Deep Learning). Possui suporte nativo para previsão probabilística e intervalos de confiança. Integração direta com modelos PyOD para detecção de anomalias. Implementado com PyTorch Lightning para modelos de Deep Learning, permitindo treinamento em GPU/TPU.

## Features

Interface unificada para modelos de previsão e detecção de anomalias; Suporte a séries temporais univariadas e multivariadas; Modelos globais treináveis em múltiplas séries; Previsão probabilística e intervalos de confiança (Conformal Prediction); Suporte a covariáveis passadas e futuras; Utilitários para backtesting e processamento de dados (escalonamento, preenchimento de valores ausentes).

## Use Cases

Previsão de demanda e vendas; Monitoramento de infraestrutura e detecção de falhas (usando detecção de anomalias); Previsão de preços de ações e criptomoedas; Análise de séries temporais em IoT e sensores; Modelagem de séries temporais hierárquicas com reconciliação.

## Integration

Instalação via `pip install darts`. Integração com PyOD para detecção de anomalias. Compatibilidade com backends como pandas, polars, numpy e xarray. Exemplo de uso de modelo estatístico:\n```python\nfrom darts import TimeSeries\nfrom darts.models import ExponentialSmoothing\n\n# Criação da série temporal\nseries = TimeSeries.from_values([10, 12, 15, 13, 18, 20])\n\n# Treinamento e previsão\nmodel = ExponentialSmoothing()\nmodel.fit(series)\nprediction = model.predict(len=3)\nprint(prediction.values())\n```

## URL

https://unit8co.github.io/darts/