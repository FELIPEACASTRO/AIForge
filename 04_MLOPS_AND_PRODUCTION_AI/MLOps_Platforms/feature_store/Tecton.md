# Tecton

## Description

Tecton é a principal **feature store para Machine Learning (ML) em tempo real** em escala, projetada para fornecer **dados atualizados em sub-segundo** e **latência de serviço inferior a 100 ms**. Criada pelos engenheiros por trás do Michelangelo da Uber, sua proposta de valor única reside na garantia de **Consistência Online/Offline** para eliminar o "train-serve skew" (desvio entre treinamento e serviço), um framework **Declarativo de Infraestrutura como Código (IaC)** para features, e um desempenho de nível empresarial com **latência P99 de sub-10 ms** e **99,99% de tempo de atividade**. A Tecton foi adquirida pela Databricks em 2025 para impulsionar a próxima geração de agentes de IA personalizados.

## Statistics

**Latência de Serviço:** Sub-10 ms (P99). **Atualização de Dados (Freshness):** 100 ms. **Disponibilidade (Uptime):** 99.99% em 100k+ QPS. **Escala:** Suporta bilhões de decisões diárias de ML em empresas Fortune 100. **Conformidade:** ISO 27001, SOC2 tipo 2 e PCI. **Impacto em Clientes:** Redução do tempo de produção de features de meses para apenas um dia; Aumento de 50% na taxa de aprovação de crédito com redução de 5% nas perdas.

## Features

**Compute Unificado e Flexível:** Suporte a Python (Ray & Arrow), Spark e SQL. **Mecanismo de Agregação de Streaming:** Agregação imediata e de ultra-baixa latência em alta escala. **Backfills de Streaming Automatizados:** Gera backfills a partir do código de feature de streaming. **Descoberta e Compartilhamento de Features:** Facilita o reuso e a governança centralizada. **Monitoramento:** Métricas de Qualidade de Dados e API de Métricas (OpenMetrics). **Integração CI/CD:** Suporte nativo para controle de versão e testes unitários.

## Use Cases

**Detecção de Fraude:** Uso de sinais comportamentais em tempo real. **Tomada de Decisão de Risco:** Decisões instantâneas com features de streaming. **Pontuação de Crédito:** Decisões de crédito precisas e em tempo real. **Personalização:** Adaptação instantânea e dinâmica de experiências de produto. **IA Agêntica (Agentic AI):** Geração e serviço de embeddings e features de contexto críticos.

## Integration

A Tecton se integra ao ecossistema de dados existente e é acessada principalmente via um cliente Python (`tecton-client`) para leitura de features em tempo real (inferência) e um SDK Python para definição de features. A plataforma é otimizada para integração com a Databricks, aproveitando o MLflow para criação e teste de endpoints de serviço.

**Exemplo de Leitura de Features (Python Client):**
```python
from tecton import get_feature_server_client
from tecton.types import FeatureService

# 1. Inicializar o cliente do Feature Server
client = get_feature_server_client(
    url="<TECTON_FEATURE_SERVER_URL>",
    api_key="<TECTON_API_KEY>"
)

# 2. Definir o Feature Service
feature_service = FeatureService(name="fraud_detection_service")

# 3. Obter features para uma entidade (ex: um usuário)
join_keys = {"user_id": "user_123"}

# 4. Chamar a API para obter as features
response = client.get_features(
    feature_service=feature_service,
    join_keys=join_keys
)

# 5. Acessar os valores das features
feature_values = response.feature_values
print(f"Valores das Features: {feature_values}")
```

## URL

https://www.tecton.ai/