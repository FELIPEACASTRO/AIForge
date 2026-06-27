# Feast - Open Source Feature Store

## Description

Feast (Feature Store) é um sistema de dados operacional de código aberto projetado para gerenciar e servir recursos de Machine Learning (ML) em escala de produção. Ele atua como uma camada de acesso a dados unificada, reutilizando a infraestrutura de dados existente para garantir a consistência dos recursos entre os ambientes de treinamento (offline) e de serviço (online), eliminando o problema crítico de 'training-serving skew'. Sua proposta de valor única reside em fornecer correção ponto-a-ponto, garantindo que os modelos sejam treinados apenas com dados que estariam disponíveis no momento da inferência, e desacoplando o ML da infraestrutura de dados subjacente [1].

## Statistics

Feast é um projeto de código aberto ativo, inicialmente desenvolvido em colaboração entre Gojek e Google [2]. Embora métricas de uso específicas de produção (como latência ou taxa de transferência) dependam da infraestrutura subjacente (por exemplo, Redis, DynamoDB para a loja online), a arquitetura do Feast é otimizada para serviço de baixa latência (sub-milissegundo) para inferência em tempo real [3].

## Features

O Feast oferece um conjunto robusto de recursos para o ciclo de vida do recurso de ML:\n\n*   **SDK Python e CLI:** Ferramentas para definir, gerenciar e interagir programaticamente com recursos.\n*   **Armazenamento Offline e Online:** Gerencia um armazenamento offline (para dados históricos de treinamento em lote) e um armazenamento online de baixa latência (para serviço de inferência em tempo real).\n*   **Correção Ponto-a-Ponto:** Lógica de junção de dados testada em batalha para evitar vazamento de dados (data leakage) durante a criação do conjunto de dados de treinamento.\n*   **Reutilização e Descoberta de Recursos:** Catálogo centralizado para definição de recursos, promovendo a colaboração entre equipes.\n*   **Servidor de Recursos (Opcional):** Um serviço hospedado para leitura e gravação de dados de recursos, útil para linguagens que não são Python [1].

## Use Cases

O Feast é amplamente aplicável em diversos domínios de ML que exigem recursos consistentes e atualizados:\n\n*   **Mecanismos de Recomendação:** Personalização de recomendações online usando recursos históricos de usuário/item e servindo recursos em tempo real.\n*   **Scorecards de Risco:** Detecção de fraude online e pontuação de crédito, usando recursos que comparam padrões históricos de transação.\n*   **NLP/RAG:** Armazenamento e indexação de vetores de embeddings de texto para pesquisa de similaridade eficiente em sistemas de Geração Aumentada por Recuperação (RAG).\n*   **Previsão de Séries Temporais:** Gerenciamento de recursos temporais e criação de agregações baseadas em tempo para previsão de demanda e detecção de anomalias [4].

## Integration

A integração com o Feast envolve a definição de recursos, a materialização de dados no armazenamento online e a recuperação de recursos para treinamento e inferência. O processo começa com a inicialização de um repositório de recursos e a definição de entidades e FeatureViews em um arquivo Python (por exemplo, `example_repo.py`).\n\n**Exemplo de Definição de Recurso (Python):**\n\n```python\nfrom feast import Entity, FeatureView, Field, FileSource, ValueType\nfrom feast.types import Float32, Int64\nfrom datetime import timedelta\n\n# 1. Definir Entidade\ndriver = Entity(name=\"driver\", description=\"ID do motorista\", value_type=ValueType.INT64)\n\n# 2. Definir Fonte de Dados\ndriver_stats_source = FileSource(\n    path=\"data/driver_stats.parquet\",\n    timestamp_field=\"event_timestamp\",\n)\n\n# 3. Definir FeatureView\ndriver_hourly_stats_fv = FeatureView(\n    name=\"driver_hourly_stats\",\n    entities=[driver],\n    ttl=timedelta(days=1),\n    schema=[\n        Field(name=\"conv_rate\", dtype=Float32),\n        Field(name=\"acc_rate\", dtype=Float32),\n        Field(name=\"avg_daily_trips\", dtype=Int64),\n    ],\n    source=driver_stats_source,\n)\n```\n\n**Exemplo de Recuperação de Recursos para Inferência (Python):**\n\n```python\nfrom feast import FeatureStore\n\n# Conectar ao Feature Store\nstore = FeatureStore(repo_path=\".\")\n\n# Recuperar recursos online para inferência em tempo real\nfeature_vector = store.get_online_features(\n    features=[\n        \"driver_hourly_stats:conv_rate\",\n        \"driver_hourly_stats:acc_rate\",\n        \"driver_hourly_stats:avg_daily_trips\",\n    ],\n    entity_rows=[\n        {\"driver_id\": 1001},\n        {\"driver_id\": 1002},\n    ],\n).to_dict()\n\nprint(feature_vector)\n# Saída esperada (exemplo): {'driver_id': [1001, 1002], 'conv_rate': [0.5, 0.8], ...}\n```

## URL

https://feast.dev/