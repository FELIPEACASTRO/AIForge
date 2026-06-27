# Comet.ml (Comet)

## Description

**Comet** é uma plataforma de avaliação de modelos de ponta a ponta para desenvolvedores, focada em rastreamento de experimentos de Machine Learning (ML) e observabilidade de sistemas de Inteligência Artificial Generativa (GenAI). Sua proposta de valor única reside na unificação do ciclo de vida de ML e GenAI, oferecendo ferramentas para rastrear, comparar, explicar e otimizar modelos, desde o treinamento até a produção. O Comet é conhecido por sua facilidade de uso, exigindo apenas algumas linhas de código para começar a rastrear experimentos, e por sua infraestrutura de nível empresarial que garante confiabilidade e segurança em escala. A plataforma se divide em duas áreas principais: **MLOps** (Gerenciamento de Experimentos, Registro de Modelos e Monitoramento de Produção) e **Opik** (uma plataforma de otimização e observabilidade de LLM de código aberto). O Comet permite que equipes de ciência de dados e engenharia acelerem o desenvolvimento de modelos, garantam a reprodutibilidade e monitorem o desempenho em produção, incluindo a detecção de desvio de dados (data drift) e o monitoramento de desempenho de LLMs. (EN: **Comet** is an end-to-end model evaluation platform for developers, focused on Machine Learning (ML) experiment tracking and Generative Artificial Intelligence (GenAI) system observability. Its unique value proposition lies in unifying the ML and GenAI lifecycle, offering tools to track, compare, explain, and optimize models, from training to production. Comet is known for its ease of use, requiring only a few lines of code to start tracking experiments, and for its enterprise-grade infrastructure that ensures reliability and security at scale. The platform is divided into two main areas: **MLOps** (Experiment Management, Model Registry, and Production Monitoring) and **Opik** (an open-source LLM observability and optimization platform). Comet enables data science and engineering teams to accelerate model development, ensure reproducibility, and monitor performance in production, including data drift detection and LLM performance monitoring.)

## Statistics

**Financiamento Total (Total Funding):** Aproximadamente $69.8 milhões [1] [2]. **Receita Anual Estimada (Estimated Annual Revenue):** Varia entre $14.9 milhões e $31.3 milhões, dependendo da fonte [1] [3]. **Usuários (Users):** Mais de 150.000 desenvolvedores e milhares de empresas confiam na plataforma [4]. **Estrelas no GitHub (GitHub Stars):** Mais de 15.430 estrelas [4]. **Equipes (Teams):** Mais de 10.000 equipes [4]. (EN: **Total Funding:** Approximately $69.8 million [1] [2]. **Estimated Annual Revenue:** Varies between $14.9 million and $31.3 million, depending on the source [1] [3]. **Users:** Over 150,000 developers and thousands of companies trust the platform [4]. **GitHub Stars:** Over 15,430 stars [4]. **Teams:** Over 10,000 teams [4].)

## Features

**Gerenciamento de Experimentos de ML (ML Experiment Management):** Rastreamento automático de código, hiperparâmetros, métricas e artefatos. Visualizações personalizadas para comparação e otimização de modelos. **Registro de Modelos (ML Model Registry):** Versionamento e gerenciamento de modelos e conjuntos de dados. **Monitoramento de Produção (ML Model Production Monitoring):** Detecção e mitigação de desvio de dados (data drift) e monitoramento de desempenho em tempo real. **Opik (Observabilidade de LLM de Código Aberto):** Rastreamento de chamadas de LLM (traces) para sistemas GenAI complexos. Debugging com feedback humano e anotação de traces. **Avaliação Automatizada de LLM:** Auto-pontuação de novas versões de aplicativos LLM com métricas para alucinação, precisão de contexto e relevância. **Otimização de Agentes de IA:** Geração e teste automatizado de prompts para sistemas agenticos. **Integração Ampla:** Suporte a frameworks de ML como PyTorch, TensorFlow, Keras, Scikit-learn, Hugging Face e plataformas de LLM como LlamaIndex e LangChain. (EN: **ML Experiment Management:** Automatic tracking of code, hyperparameters, metrics, and artifacts. Custom visualizations for model comparison and optimization. **ML Model Registry:** Versioning and management of models and datasets. **ML Model Production Monitoring:** Detection and mitigation of data drift and real-time performance monitoring. **Opik (Open-Source LLM Observability):** Tracking of LLM calls (traces) for complex GenAI systems. Debugging with human feedback and trace annotation. **Automated LLM Evaluation:** Auto-scoring of new LLM application versions with metrics for hallucination, context precision, and relevance. **AI Agent Optimization:** Automated prompt generation and testing for agentic systems. **Broad Integration:** Support for ML frameworks like PyTorch, TensorFlow, Keras, Scikit-learn, Hugging Face, and LLM platforms like LlamaIndex and LangChain.)

## Use Cases

**Aceleração do Ciclo de Vida de ML:** Cientistas de dados utilizam o Comet para rastrear e comparar milhares de experimentos, garantindo a reprodutibilidade e acelerando o ciclo de pesquisa e desenvolvimento de modelos. **Garantia de Conformidade e Auditoria:** O registro detalhado de cada experimento, incluindo código, dados e resultados, facilita a conformidade regulatória e a auditoria de modelos em ambientes empresariais. **Monitoramento de Modelos em Produção:** Empresas utilizam o monitoramento de produção do Comet para detectar desvios de dados (data drift) e degradação de desempenho em tempo real, permitindo intervenção rápida para manter a precisão do modelo. **Observabilidade e Debugging de LLMs:** Desenvolvedores de GenAI usam o Opik para visualizar e depurar sistemas complexos baseados em LLM, rastreando a cadeia de chamadas (traces) e identificando problemas como alucinações ou imprecisão de contexto. **Otimização de Agentes de IA:** O Opik é aplicado para otimizar o desempenho de agentes de IA, automatizando a geração e o teste de prompts para encontrar as configurações de melhor desempenho. (EN: **ML Lifecycle Acceleration:** Data scientists use Comet to track and compare thousands of experiments, ensuring reproducibility and accelerating the model research and development cycle. **Compliance and Audit Assurance:** The detailed logging of every experiment, including code, data, and results, facilitates regulatory compliance and model auditing in enterprise environments. **Production Model Monitoring:** Companies use Comet's production monitoring to detect data drift and performance degradation in real-time, allowing for quick intervention to maintain model accuracy. **LLM Observability and Debugging:** GenAI developers use Opik to visualize and debug complex LLM-based systems, tracing the call chain and identifying issues like hallucinations or context inaccuracy. **AI Agent Optimization:** Opik is applied to optimize the performance of AI agents, automating prompt generation and testing to find the best-performing configurations.)

## Integration

A integração do Comet é notavelmente simples, exigindo apenas a instalação da biblioteca Python e a inicialização de um objeto `Experiment` ou o uso do decorador `@track` do Opik. O Comet se integra automaticamente com a maioria dos frameworks de ML e LLM.

**Exemplo de Integração com PyTorch (ML Experiment Tracking):**

```python
import torch
from comet_ml import Experiment

# 1. Inicialize o experimento
experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="meu-projeto-pytorch",
    workspace="meu-workspace"
)

# 2. Registre hiperparâmetros
hyper_params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}
experiment.log_parameters(hyper_params)

# 3. Seu código de treinamento PyTorch
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])

for epoch in range(hyper_params["epochs"]):
    # ... (código de treinamento)
    loss = torch.rand(1) # Exemplo de cálculo de perda
    
    # 4. Registre métricas
    experiment.log_metric("loss", loss.item(), step=epoch)

# 5. Registre o modelo final
experiment.log_model("meu_modelo_pytorch", model)
experiment.end()
```

**Exemplo de Integração com Opik (LLM Observability):**

```python
from opik import track

# Use o decorador @track para rastrear automaticamente as chamadas
@track
def llm_chain(user_question):
    context = get_context(user_question)
    response = call_llm(user_question, context)
    return response

@track
def get_context(user_question):
    # Lógica de recuperação de contexto
    return ["O cachorro perseguiu o gato.", "O gato se chamava Luky."]

@track
def call_llm(user_question, context):
    # Chamada para o LLM (pode ser combinado com qualquer integração Opik)
    return "O cachorro perseguiu o gato Luky."

response = llm_chain("O que o cachorro fez?")
print(response)
```
(EN: Comet's integration is remarkably simple, requiring only the installation of the Python library and the initialization of an `Experiment` object or the use of the Opik `@track` decorator. Comet automatically integrates with most ML and LLM frameworks. **PyTorch Integration Example (ML Experiment Tracking):** [Code above] **Opik Integration Example (LLM Observability):** [Code above])

## URL

https://www.comet.com/site/