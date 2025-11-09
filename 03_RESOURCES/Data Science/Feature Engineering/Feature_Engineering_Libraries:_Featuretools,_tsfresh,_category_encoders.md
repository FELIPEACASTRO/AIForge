# Feature Engineering Libraries: Featuretools, tsfresh, category_encoders

## Description

**Featuretools** é uma biblioteca Python de código aberto para **Engenharia de Recursos Automatizada (AFE)**. Sua principal proposta de valor é simplificar o processo de criação de recursos a partir de dados relacionais e temporais, permitindo que cientistas de dados se concentrem na modelagem. Ele utiliza uma técnica chamada **Deep Feature Synthesis (DFS)** para gerar automaticamente centenas de recursos a partir de múltiplas tabelas de dados, eliminando a necessidade de codificação manual e demorada. A biblioteca também se destaca por sua capacidade de gerar descrições em linguagem natural para os recursos criados, tornando o processo mais transparente e interpretável.

**Featuretools** is an open-source Python library for **Automated Feature Engineering (AFE)**. Its core value proposition is to simplify the process of creating features from relational and temporal data, allowing data scientists to focus on modeling. It uses a technique called **Deep Feature Synthesis (DFS)** to automatically generate hundreds of features from multiple data tables, eliminating the need for time-consuming manual coding. The library is also notable for its ability to generate natural language descriptions for the created features, making the process more transparent and interpretable.

## Statistics

**Featuretools:**
*   **GitHub Stars:** 7.562
*   **Downloads PyPI (Geral):** 752.916

**tsfresh:**
*   **GitHub Stars:** 9.011
*   **Downloads PyPI (Geral):** 1.919.582

**category_encoders:**
*   **GitHub Stars:** 2.467
*   **Downloads PyPI (Geral):** 12.615.710

## Features

**Featuretools:** Deep Feature Synthesis (DFS), Suporte a dados relacionais e temporais (EntitySets), Geração automática de recursos, Descrições de recursos em linguagem natural, Primitivas de recursos personalizáveis, Integração com Dask e Spark para escalabilidade.

**tsfresh:** Extração de mais de 100 recursos de séries temporais, Seleção automática de recursos relevantes (baseada em testes estatísticos), Suporte a séries temporais multivariadas, Integração com pandas DataFrames, Capacidade de lidar com séries temporais com diferentes comprimentos.

**category_encoders:** Implementação de mais de 15 métodos de codificação categórica (incluindo TargetEncoder, BinaryEncoder, HashingEncoder, etc.), Interface compatível com scikit-learn (fit/transform), Suporte a pandas DataFrames, Lida com variáveis de alta cardinalidade de forma eficiente.

## Use Cases

**Featuretools:** Previsão de rotatividade de clientes (Churn Prediction), Detecção de fraudes em transações financeiras, Sistemas de recomendação, Previsão de falhas de equipamentos (Manutenção Preditiva) em dados de sensores e logs.

**tsfresh:** Classificação e agrupamento de séries temporais (por exemplo, em dados de sensores, sinais médicos como ECG, ou dados financeiros), Análise de falhas de máquinas baseada em vibração, Análise de padrões de uso de energia.

**category_encoders:** Qualquer problema de Machine Learning que envolva variáveis categóricas (por exemplo, modelos de regressão e classificação), especialmente em cenários com **alta cardinalidade** (muitos valores únicos) onde o One-Hot Encoding se torna ineficiente. Comum em competições de ciência de dados (Kaggle).

## Integration

**Featuretools:**
Instalação: `pip install featuretools`
Exemplo de uso (DFS):
```python
import featuretools as ft
import pandas as pd

# 1. Criar EntitySet
data = {'clientes': pd.DataFrame({'id': [1, 2, 3], 'idade': [30, 40, 50]}),
        'transacoes': pd.DataFrame({'id': [1, 2, 3, 4], 'cliente_id': [1, 1, 2, 3], 'valor': [10, 20, 5, 15]})}
es = ft.EntitySet(id="dados_comerciais")
es = es.add_dataframe(dataframe_name="clientes", dataframe=data['clientes'], index="id")
es = es.add_dataframe(dataframe_name="transacoes", dataframe=data['transacoes'], index="id")
es = es.add_relationship(ft.Relationship(es["clientes"]["id"], es["transacoes"]["cliente_id"]))

# 2. Executar Deep Feature Synthesis (DFS)
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="clientes", max_depth=2)
print(feature_matrix)
```

**tsfresh:**
Instalação: `pip install tsfresh`
Exemplo de uso (Extração de recursos):
```python
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
import pandas as pd

# Dados de exemplo (formato longo)
df_ts = pd.DataFrame({'id': [1, 1, 1, 2, 2], 'tempo': [1, 2, 3, 1, 2], 'valor': [10, 15, 12, 5, 8]})

# Extrair recursos (usando um conjunto mínimo de parâmetros)
settings = MinimalFCParameters()
features = extract_features(df_ts, column_id="id", column_sort="tempo", column_value="valor", default_fc_parameters=settings)
print(features)
```

**category_encoders:**
Instalação: `pip install category-encoders`
Exemplo de uso (Target Encoding):
```python
import category_encoders as ce
import pandas as pd

# Dados de exemplo
data = pd.DataFrame({'cor': ['vermelho', 'azul', 'verde', 'vermelho', 'azul'], 'alvo': [1, 0, 1, 1, 0]})
X = data['cor']
y = data['alvo']

# Aplicar Target Encoder
encoder = ce.TargetEncoder(cols=['cor'])
X_encoded = encoder.fit_transform(X, y)
print(X_encoded)
```

## URL

Featuretools: https://featuretools.alteryx.com/ | tsfresh: https://tsfresh.readthedocs.io/ | category_encoders: https://contrib.scikit-learn.org/category_encoders/