# Arize AI - ML Observability and Monitoring

## Description

A Arize AI é uma **Plataforma de Engenharia de Agentes e IA** unificada, projetada para observabilidade e avaliação de modelos de Machine Learning (ML) e Agentes de IA em produção. Sua proposta de valor única é fechar o ciclo entre o desenvolvimento e a produção de IA, permitindo um ciclo de iteração orientado por dados. A plataforma, conhecida como **Arize AX**, capacita organizações a gerenciar e melhorar ofertas de IA em escala, garantindo que a observabilidade de produção se alinhe com avaliações confiáveis. A Arize AI atende tanto a modelos tradicionais de ML e Visão Computacional quanto à crescente área de IA Generativa e LLMs.

## Statistics

A plataforma Arize AI processa grandes volumes de dados de produção de IA, com as seguintes métricas chave:
*   **1 Trilhão** de spans por mês (rastreamento de agentes e LLMs).
*   **50 Milhões** de avaliações por mês.
*   **5 Milhões** de downloads por mês (referente ao Phoenix OSS, a biblioteca de código aberto para LLM Tracing e Avaliação).

## Features

A plataforma Arize AI (Arize AX) oferece observabilidade para IA Generativa e ML & Visão Computacional, cobrindo três pilares principais:

**1. Desenvolvimento:**
*   **Otimização de Prompt:** Agentes auto-aprimoráveis com otimização automática baseada em avaliações e anotações.
*   **Replay no Playground:** Ferramenta para depuração e aperfeiçoamento de prompts.
*   **Serviço e Gerenciamento de Prompt:** Gerenciamento centralizado de prompts e serviço rápido de otimizações.

**2. Avaliação:**
*   **Experimentos CI/CD:** Detecção precoce de regressões de prompt e agente através de CI/CD orientado por avaliação.
*   **LLM como Juiz (LLM as a Judge):** Avaliação automática e em escala de prompts e ações de agentes.
*   **Anotação Humana e Filas:** Gerenciamento de filas de rotulagem, anotações de produção e criação de conjuntos de dados dourados.

**3. Observabilidade:**
*   **Rastreamento de Padrão Aberto (Open Standard Tracing):** Rastreamento de agentes e frameworks alimentado por OpenTelemetry (OTEL).
*   **Avaliações Online (Online Evals):** Captura instantânea de problemas com IA avaliando IA.
*   **Monitoramento e Dashboards:** Monitoramento de IA em tempo real com dashboards analíticos avançados.

## Use Cases

A Arize AI é utilizada para garantir a confiabilidade e o desempenho de modelos de IA e agentes em produção, abrangendo tanto a IA Generativa quanto o ML tradicional e Visão Computacional.

*   **Agentes de IA:** Monitoramento e depuração de agentes de IA em tempo real, como em análise de dados de frota (Geotab) e reservas de viagens (Priceline).
*   **Mitigação de Viés:** Exame de modelos de pontuação de crédito para detectar e mitigar viés, garantindo decisões justas e livres de discriminação.
*   **Avaliação de LLM (LLM Evals):** Aplicação de avaliações em tempo real para modelos de linguagem grande (LLMs), como nos exemplos da Bazaarvoice, para medir métricas como relevância, taxa de alucinação e latência.
*   **Setores Diversos:** Soluções para Visão Computacional, Previsão (Forecasting), Serviços Financeiros e Manufatura.
*   **Desenvolvimento de Copiloto:** Uso da plataforma para desenvolver, iterar e aprimorar assistentes de IA (Copilots).

## Integration

A integração com a Arize AI é realizada principalmente através de seus SDKs e bibliotecas, com foco em Python.

**1. Arize AX (ML Observability) - Python SDK:**
O SDK Python é a principal ferramenta para registrar dados de modelos de Machine Learning (previsões, rótulos reais, features, etc.) para monitoramento.

```python
# Exemplo Conceitual de Uso do SDK Python (Arize AX)
from arize.api import Client

# Inicializar o cliente com a chave API e Hostname
arize_client = Client(
    space_key="YOUR_SPACE_KEY",
    api_key="YOUR_API_KEY",
    host="YOUR_HOST"
)

# Registrar dados de previsão para um modelo
arize_client.log_prediction(
    model_id="meu-modelo-de-credito",
    prediction_id="id-unico-da-predicao",
    prediction_label="alto_risco",
    actual_label="alto_risco",
    features={"idade": 35, "renda": 50000},
    # ... outros parâmetros como embeddings, métricas, etc.
)
```

**2. Arize Phoenix (LLM Tracing e Avaliação) - Open Source:**
O Phoenix é uma biblioteca de código aberto para rastreamento e avaliação de LLMs, que se integra ao Arize AX.

```python
# Exemplo Conceitual de Rastreamento com Phoenix (Python)
import phoenix as px
from phoenix.trace import Span

# Iniciar o rastreador Phoenix
px.start_session(
    project_name="meu-projeto-llm",
    host="YOUR_PHOENIX_HOST" # Pode ser o host local ou o Arize AX
)

# Exemplo de rastreamento manual de um span
with Span(name="geracao_resposta", context={"prompt": "..."}) as span:
    # Lógica de chamada do LLM
    response = "Resposta do LLM"
    span.log_attributes({"response": response})
# O rastreamento também suporta integrações automáticas com frameworks como OpenAI, LangChain, etc.
```

## URL

https://arize.com/