# Darts

## Description

Darts is an open-source Python library for time series forecasting and anomaly detection, designed to be user-friendly and unified. It offers a wide range of models, from classics such as ARIMA to deep neural networks (N-BEATS), all with a consistent `fit()` and `predict()` interface similar to scikit-learn. Its unique value proposition lies in its ease of use, native support for multivariate time series, and the ability to train global models on multiple datasets.

## Statistics

Supports more than 20 forecasting models (statistical and Deep Learning). Provides native support for probabilistic forecasting and confidence intervals. Direct integration with PyOD models for anomaly detection. Implemented with PyTorch Lightning for Deep Learning models, enabling training on GPU/TPU.

## Features

Unified interface for forecasting and anomaly detection models; Support for univariate and multivariate time series; Global models trainable on multiple series; Probabilistic forecasting and confidence intervals (Conformal Prediction); Support for past and future covariates; Utilities for backtesting and data processing (scaling, filling missing values).

## Use Cases

Demand and sales forecasting; Infrastructure monitoring and fault detection (using anomaly detection); Forecasting stock and cryptocurrency prices; Time series analysis in IoT and sensors; Hierarchical time series modeling with reconciliation.

## Integration

Installation via `pip install darts`. Integration with PyOD for anomaly detection. Compatibility with backends such as pandas, polars, numpy, and xarray. Example of using a statistical model:\n```python\nfrom darts import TimeSeries\nfrom darts.models import ExponentialSmoothing\n\n# Creating the time series\nseries = TimeSeries.from_values([10, 12, 15, 13, 18, 20])\n\n# Training and forecasting\nmodel = ExponentialSmoothing()\nmodel.fit(series)\nprediction = model.predict(len=3)\nprint(prediction.values())\n```

## URL

https://unit8co.github.io/darts/