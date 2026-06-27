# Google Vertex AI

## Description

O Google Vertex AI é uma plataforma de desenvolvimento de IA unificada e totalmente gerenciada do Google Cloud, projetada para simplificar e acelerar o ciclo de vida de Machine Learning (ML) e IA Generativa (GenAI) de ponta a ponta. Sua proposta de valor única reside na unificação de todos os serviços de ML em uma única interface e API, permitindo que cientistas de dados e engenheiros de ML automatizem, padronizem e gerenciem projetos de ML com eficiência. A plataforma se destaca pela integração nativa com os modelos Gemini e pelo acesso a mais de 200 modelos de fundação (via Model Garden), aprimorando significativamente suas capacidades de GenAI de nível empresarial.

## Statistics

**Redução de Tempo de Implantação:** Visa reduzir o tempo necessário para treinar e implantar modelos em produção em até 80% devido à unificação dos serviços. **Modelos de Fundação:** Oferece acesso a mais de 200 modelos de fundação no Model Garden, incluindo modelos do Google (Gemini, Imagen) e de terceiros/código aberto (Llama 3.2, Claude). **Métricas de Monitoramento:** Exporta métricas detalhadas para o Cloud Monitoring, como Utilização de CPU/Memória e Utilização de Memória do Acelerador (GPU/TPU).

## Features

**Plataforma Unificada:** Unifica todos os serviços de ML do Google Cloud em uma única interface e API. **IA Generativa:** Acesso nativo aos modelos Gemini e a mais de 200 modelos de fundação (incluindo terceiros e código aberto) via Model Garden, com ferramentas como Vertex AI Studio e Agent Builder. **MLOps Completo:** Conjunto robusto de ferramentas para o ciclo de vida de MLOps, incluindo Vertex AI Pipelines (orquestração), Model Registry (gerenciamento), Feature Store (serviço de recursos) e Vertex AI Evaluation (avaliação de modelos). **Desenvolvimento Flexível:** Suporta AutoML (treinamento sem código), Treinamento Personalizado (com frameworks preferidos) e Vertex AI Workbench/Colab Enterprise (notebooks integrados ao BigQuery).

## Use Cases

**IA Generativa:** Criação de agentes de atendimento ao cliente (ex: LUXGEN), geração de conteúdo (texto, imagem, código) e sumarização de documentos. **ML Preditivo:** Análise de Sentimento em Escala, Previsão de Vendas (usando modelos ARIMA e LSTM) e Classificação de Imagens Personalizada. **MLOps e Governança:** Orquestração de pipelines de ML de ponta a ponta para reprodutibilidade e automação, e monitoramento de modelos em produção para desvio de dados (*data drift*) e enviesamento (*skew*).

## Integration

A principal integração é realizada através do **Vertex AI SDK para Python** (`google-cloud-aiplatform`). O SDK permite a automação de tarefas como treinamento, implantação e previsão. A plataforma também se integra nativamente com outros serviços do Google Cloud, como o **BigQuery** (para armazenamento de dados e como *offline store* para o Feature Store) e o **Cloud Monitoring** (para métricas de desempenho).

**Exemplo de Implantação de Modelo (Python SDK):**
```python
from google.cloud import aiplatform

# Inicializar o cliente com projeto e região
aiplatform.init(project='YOUR_PROJECT_ID', location='YOUR_REGION')

# Obter o modelo treinado
model = aiplatform.Model.list(filter='display_name="my-trained-model"')[0]

# Criar e implantar o modelo em um endpoint
endpoint = aiplatform.Endpoint.create(display_name='my-model-endpoint')
model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-2',
    min_replica_count=1,
    max_replica_count=1
)
```

## URL

https://cloud.google.com/vertex-ai