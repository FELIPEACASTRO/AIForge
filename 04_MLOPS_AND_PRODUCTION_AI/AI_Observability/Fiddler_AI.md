# Fiddler AI

## Description

**Fiddler AI** é uma plataforma unificada de **Observabilidade de IA** (AI Observability) que se estende além do gerenciamento tradicional de desempenho de modelos de Machine Learning (ML). A plataforma fornece visibilidade e insights acionáveis para monitorar, analisar, explicar e proteger modelos de ML, LLMs (Large Language Models) e sistemas agenticos em produção. Sua proposta de valor única reside na capacidade de oferecer **transparência e confiança** em todo o ciclo de vida da IA, desde a avaliação até a produção, por meio de recursos avançados como detecção de desvio (drift), análise de causa raiz e explicabilidade (Explainable AI - XAI). A Fiddler permite que equipes de Data Science e MLOps operacionalizem modelos em escala, garantindo que o desempenho, a justiça e a conformidade sejam mantidos de forma contínua. A plataforma evoluiu de Model Performance Management (MPM) para uma solução completa de Observabilidade de IA, abrangendo a nova fronteira de sistemas de IA generativa e agenticos.

## Statistics

A Fiddler AI monitora uma ampla gama de métricas, que podem ser categorizadas em: **Métricas de Desempenho de ML:** *Accuracy, Precision, Recall, F1-score, AUC-ROC* (para classificação), *RMSE, MAE* (para regressão). **Métricas de Qualidade de Dados e Desvio:** *Data Drift* (desvio de dados), *Model Drift* (desvio de modelo), *Feature Importance Drift*. **Métricas de Negócios e Operacionais:** *Latência, Taxas de Erro, Throughput*. **Métricas de LLM e Agentic:** Mais de 80 métricas prontas para uso para **LLM Observability** e **Agentic Observability**, incluindo métricas de segurança, toxicidade, alucinação, e desempenho de *Guardrails* e *Trust Models*. Um caso de uso notável é a **Marinha dos EUA**, que reduziu em 97% o tempo necessário para atualizar o modelo de IA usando a plataforma Fiddler.

## Features

**Observabilidade Unificada de IA:** Monitoramento de modelos de ML, LLMs e sistemas agenticos em uma única plataforma. **Monitoramento de Modelos (Model Monitoring):** Detecção de desvio de dados (data drift), desvio de modelo (model drift), anomalias, latência e taxas de erro. **Explicabilidade de IA (Explainable AI - XAI):** Fornece insights sobre o "porquê" e o "como" das decisões do modelo, incluindo análise de causa raiz. **Guardrails e Trust Models:** Oferece mais de 80 métricas prontas para uso e suporte para métricas personalizadas para garantir a segurança e a conformidade da IA. **Analytics:** Conecta previsões de modelos com contexto de negócios para insights acionáveis. **Detecção de Viés (Bias Detection) e Fairness:** Ferramentas para mitigar viés e construir sistemas de IA responsáveis. **Suporte a Modelos Complexos:** Inclui monitoramento de modelos de Processamento de Linguagem Natural (NLP) e Visão Computacional (CV).

## Use Cases

**Serviços Financeiros:** Garantir empréstimos e negociações transparentes e justos, mitigando o viés em modelos de risco de crédito. **Governo e Defesa:** Como a **Marinha dos EUA**, para gerenciar e atualizar modelos de IA em ambientes críticos com alta conformidade e segurança. **E-commerce e Varejo:** Otimizar a experiência do cliente e estender o valor vitalício do cliente (LTV) por meio de modelos de recomendação e precificação monitorados. **Saúde:** Melhorar os resultados dos pacientes com observabilidade de IA em modelos de diagnóstico e tratamento. **MLOps e Data Science:** Fornecer às equipes uma plataforma unificada para acelerar a operacionalização de modelos de ML em escala, desde a experimentação até a produção, garantindo que os modelos permaneçam confiáveis e em conformidade. **Sistemas Agenticos e LLMs:** Monitorar, analisar e proteger agentes de IA e aplicações de LLM, garantindo que as interações sejam seguras, precisas e dentro das políticas de uso.

## Integration

A Fiddler AI oferece um **SDK Python** robusto para integração com ambientes de MLOps, como Jupyter Notebooks e pipelines automatizados. A integração é tipicamente realizada em duas etapas principais: **1. Onboarding do Modelo:** O modelo é registrado na plataforma Fiddler, definindo seu esquema (`ModelSpec`), tipo de tarefa (`ModelTask`) e fornecendo um *sample dataframe* para inferência de esquema. **2. Publicação de Dados de Inferência:** Dados de produção (inferência) são publicados na Fiddler em tempo real ou em lotes para monitoramento contínuo. A plataforma se integra nativamente com ecossistemas de ML populares, como **Amazon SageMaker**, **Google Vertex AI**, e **Databricks** (via MLflow).

**Exemplo de Código Python (Simplificado para Onboarding):**

```python
import fiddler as fdl
import pandas as pd

# 1. Conectar ao ambiente Fiddler (credenciais e URL omitidas)
# client = fdl.FiddlerClient(url=FIDDLER_URL, org_id=ORG_ID, auth_token=AUTH_TOKEN)

PROJECT_NAME = 'quickstart_example'
MODEL_NAME = 'my_model'

# Criar um projeto
# project = fdl.Project(name=PROJECT_NAME)
# project.create()

# Definir o ModelSpec (função de cada coluna)
model_spec = fdl.ModelSpec(
    inputs=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],
    outputs=['probability_churned'],
    targets=['Churned'],
    decisions=[],
    metadata=[],
    custom_features=[],
)

# Definir a tarefa do modelo (ex: Classificação Binária)
model_task = fdl.ModelTask.BINARY_CLASSIFICATION
task_params = fdl.ModelTaskParams(target_class_order=['no', 'yes'])

# Criar um DataFrame de exemplo (sample_df)
# sample_df = pd.read_csv('path/to/sample_data.csv')

# Onboard do modelo
# model = fdl.Model.from_data(
#     name=MODEL_NAME,
#     project_id=fdl.Project.from_name(PROJECT_NAME).id,
#     source=sample_df,
#     spec=model_spec,
#     task=model_task,
#     task_params=task_params,
#     event_id_col='id_column',
#     event_ts_col='timestamp_column'
# )
```

## URL

https://www.fiddler.ai/