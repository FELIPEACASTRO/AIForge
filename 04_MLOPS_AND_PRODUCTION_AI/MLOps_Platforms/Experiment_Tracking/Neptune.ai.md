# Neptune.ai

## Description

Neptune.ai é um **rastreador de experimentos (experiment tracker) e um repositório de metadados (metadata store) para MLOps**, projetado especificamente para o treinamento e depuração de **modelos de fundação (foundation models)** em grande escala. Sua proposta de valor única reside na capacidade de monitorar milhares de métricas por camada (per-layer metrics) — perdas, gradientes e ativações — em qualquer escala, com visualização sem atrasos e renderização 100% precisa, mesmo com milhões de pontos de dados. Foca em ser uma ferramenta dedicada e não-intrusiva, integrando-se facilmente com stacks de ML existentes, e oferece opções de hospedagem SaaS ou auto-hospedada (self-hosted) via Helm chart para Kubernetes.

## Statistics

**Usuários:** Mais de **60.000 pesquisadores de IA** e **1.500 equipes comerciais e de pesquisa** utilizam a plataforma. **Projetos:** Mais de **30.000 projetos** rastreados. **Reconhecimento:** Incluído na lista "Top 100 AI Startups" da CB Insights em 2021 e 2022. **Avaliação:** Classificação de **4.8 de 5 estrelas** em avaliações de usuários (G2). **Financiamento:** Apoiado por **$18 milhões** em financiamento. **Escala de Dados:** Demonstrações públicas com mais de **100 milhões de pontos de dados** por execução.

## Features

**Rastreamento de Experimentos Escalável:** Log, exibição, organização e comparação de experimentos de ML em um único lugar, otimizado para modelos de fundação e LLMs. **Repositório de Metadados Centralizado:** Armazenamento de metadados de MLOps, incluindo scripts, dados, parâmetros e métricas. **Depuração Profunda:** Facilita a identificação de problemas de treinamento (gradientes explosivos/evanescentes, falhas de convergência) isolando problemas em camadas específicas. **Forking de Runs:** Permite testar múltiplas configurações simultaneamente e ramificar a partir do melhor passo final, mantendo a linhagem completa do experimento. **Registro de Modelos (Model Registry):** Funcionalidade para versionar, armazenar e organizar metadados de modelos. **Visualização de Alto Desempenho:** Filtragem e pesquisa rápidas de dados, visualização e comparação de métricas, parâmetros e curvas de aprendizado em tempo real.

## Use Cases

**Treinamento de Modelos de Fundação:** Monitoramento e depuração de modelos de linguagem grandes (LLMs) e outros modelos de fundação em escala massiva (ex: OpenAI, Bioptimus). **Otimização de Hiperparâmetros:** Agrupamento, filtragem e classificação de milhares de experimentos para insights claros e tomada de decisão confiante. **Pesquisa e Desenvolvimento de IA:** Fornece um registro transparente e pesquisável do trabalho, essencial para ciência aplicada rigorosa (ex: KoBold Metals). **MLOps e Gerenciamento de Modelos:** Integração em pipelines de MLOps para gerenciamento estruturado e segurança aprimorada (ex: Veo Technologies, Cradle). **Empresas Notáveis:** OpenAI, Samsung, Roche, HP, Brainly, Ginkgo Bioworks, Cradle, KoBold Metals, InstaDeep, Bioptimus.

## Integration

Neptune.ai integra-se com várias bibliotecas de machine learning (PyTorch, TensorFlow, Keras, Scikit-learn, etc.) através de loggers ou callbacks. O uso principal envolve a inicialização de uma "run" e o log de configurações e métricas. A integração utiliza as bibliotecas `neptune-scale` para logging e `neptune-query` para consulta.

**Exemplo de Integração (Python):**

```python
# 1. Instalação das bibliotecas
# pip install neptune-scale neptune-query

# 2. Conexão e Criação de uma Run (neptune-scale)
from neptune_scale import Run

run = Run(
    run_id="MY-PROJECT-123", # ID do projeto e da run
    experiment_name="foundation-model-training"
)

# 3. Log de Hiperparâmetros e Processo de Treinamento
run.log_configs(
    {
        "params/learning_rate": 0.001,
        "params/optimizer": "Adam",
    }
)

# Loop de treinamento
for step in range(100):
    # Simulação de log de métricas
    accuracy = 0.87 + (step / 1000)
    loss = 0.14 - (step / 5000)
    
    run.log_metrics(
        data={
            "train/accuracy": accuracy,
            "train/loss": loss,
        },
        step=step,
    )

# 4. Consulta de Logs para Análise (neptune-query)
import neptune_query as nq

# Buscar metadados como tabela (DataFrame)
# table = nq.fetch_experiments_table(
#     experiments=r"foundation-model-.*",
#     attributes=r".*metric.*/val_.+",
# )
# print(table.head())

# Finalizar a run
run.stop()
```

## URL

https://neptune.ai/