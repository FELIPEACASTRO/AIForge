# IBM watsonx.ai (Evolução do IBM Watson Studio)

## Description

O IBM watsonx.ai é a evolução do IBM Watson Studio, sendo um estúdio de desenvolvimento de IA integrado, de ponta a ponta, projetado para simplificar e escalar a construção e a implementação de modelos de IA, incluindo modelos de base e IA generativa. Sua proposta de valor única reside em ser uma plataforma unificada que oferece acesso a modelos de fundação confiáveis (como o IBM Granite, modelos de terceiros e modelos de código aberto via Model Gateway), permitindo que as empresas se concentrem na aplicação de IA para resultados de negócios. Ele fornece uma experiência de desenvolvimento colaborativa, com ou sem código, para todo o ciclo de vida da IA, desde a preparação de dados até a implementação e monitoramento, e pode ser implantado em ambientes de nuvem híbrida (SaaS ou auto-hospedado no IBM Cloud Pak for Data).

## Statistics

**Resultados de Clientes (watsonx.ai):** AddAI viu 50% menos consultas de atendimento ao cliente não respondidas; Silver Egg Technology antecipa um processo de contratação 75% mais rápido; UHCW NHS Trust atendeu 700 pacientes a mais semanalmente sem adicionar pessoal; Blendow Group viu 90% menos tempo necessário para resumir e analisar documentos. **Métricas de Suporte (G2):** watsonx.ai supera o Watson Studio com uma pontuação de 8.8 em qualidade de suporte.

## Features

Model Gateway (acesso a modelos de fundação como Granite, terceiros e open source); Full AI Lifecycle Management (gerenciamento completo do ciclo de vida da IA); Developer AI Toolkit (SDKs, APIs, fluxos de trabalho agentivos, frameworks RAG); Content and Code Generation; Knowledge Management (com RAG templates); Insight Extraction and Forecasting; Suporte a ambientes Híbridos (Cloud e on-premises).

## Use Cases

**Atendimento ao Cliente:** Redução de consultas não respondidas e melhoria da satisfação. **Recursos Humanos:** Aceleração do processo de contratação. **Saúde:** Otimização da capacidade de atendimento ao paciente. **Análise de Documentos:** Resumo e análise eficientes de grandes volumes de documentos. **Desenvolvimento de Agentes de IA:** Criação de assistentes e agentes de IA para automatizar processos de negócios.

## Integration

A integração é feita principalmente através do **IBM watsonx.ai Python SDK** (`ibm-watsonx-ai`) e APIs REST.

**Exemplo de Inferência de Texto (Python SDK):**
```python
from ibm_watsonx_ai.client import Client
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# 1. Configuração do Cliente
# Substitua 'YOUR_API_KEY' e 'YOUR_PROJECT_ID'
client = Client(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="YOUR_API_KEY",
    project_id="YOUR_PROJECT_ID"
)

# 2. Definição de Parâmetros
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 50,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.REPETITION_PENALTY: 1
}

# 3. Prompt e Modelo
prompt = "Explique o conceito de IA generativa em uma frase."
model_id = ModelTypes.GRANITE_13B_INSTRUCT

# 4. Geração de Texto
response = client.model.generate(
    model_id=model_id,
    prompt=prompt,
    params=parameters
)

# 5. Resultado
generated_text = response['results'][0]['generated_text']
# print(generated_text)
```

## URL

https://www.ibm.com/products/watsonx-ai