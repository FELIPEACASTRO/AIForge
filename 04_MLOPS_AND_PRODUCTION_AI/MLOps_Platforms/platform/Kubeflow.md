# Kubeflow

## Description

O Kubeflow é uma **plataforma de Machine Learning (ML) de código aberto e nativa do Kubernetes**, dedicada a tornar as implantações de fluxos de trabalho de ML simples, portáteis e escaláveis. Sua proposta de valor única reside em fornecer uma **plataforma unificada e modular** para todo o ciclo de vida do ML (MLOps), desde a experimentação até a produção, aproveitando a orquestração, escalabilidade e portabilidade inerentes do Kubernetes. Ele permite que cientistas de dados e engenheiros de ML construam, treinem e implantem modelos em qualquer infraestrutura onde o Kubernetes possa ser executado (nuvem pública, nuvem privada, on-premises).

## Statistics

*   **Adoção em Produção:** 44% dos usuários relataram estar executando o Kubeflow em produção (Pesquisa de Usuários de 2022).
*   **Uso de Múltiplos Componentes:** 84% a 85% dos usuários utilizam mais de um componente do Kubeflow, indicando a adoção da plataforma completa e modular.
*   **Componentes Mais Usados (Pesquisa de 2023):** Pipelines (90%), Notebooks (76%), e Katib (47%).
*   **Comunidade:** Grande e ativa, com uso por empresas da Global 500.

## Features

O Kubeflow é composto por vários componentes modulares que cobrem o ciclo de vida completo do ML:
*   **Kubeflow Pipelines (KFP):** Criação e orquestração de fluxos de trabalho de ML portáteis e escaláveis, baseados em contêineres.
*   **Notebooks:** Spawning e gerenciamento de instâncias de Jupyter Notebooks (e outros) no Kubernetes para experimentação e desenvolvimento.
*   **Katib:** Serviço de ajuste de hiperparâmetros e busca neural (AutoML) para otimizar modelos.
*   **Training Operators:** Treinamento distribuído de modelos de ML usando frameworks populares como TensorFlow, PyTorch, MXNet e XGBoost.
*   **KFServing (KServe):** Implantação, serviço e gerenciamento de modelos de ML em escala, com recursos como auto-escalonamento, canary rollouts e A/B testing.
*   **Metadata:** Rastreamento e gerenciamento de metadados de artefatos de ML (datasets, modelos, pipelines).

## Use Cases

*   **Criação de Plataformas MLOps:** Empresas usam o Kubeflow como a espinha dorsal para construir suas próprias plataformas internas de MLOps, padronizando o desenvolvimento e a implantação de ML.
*   **Treinamento Distribuído:** Execução de tarefas de treinamento de modelos de grande escala que exigem múltiplos GPUs ou CPUs, como o treinamento de modelos de Linguagem Grande (LLMs) ou modelos de Visão Computacional.
*   **Experimentação e Reprodução:** Uso de Notebooks e Pipelines para garantir que os experimentos de ML sejam reproduzíveis e possam ser facilmente movidos da fase de P&D para a produção.
*   **Serviço de Modelo em Escala:** Implantação de modelos de ML como microsserviços escaláveis com o KServe, permitindo inferência de baixa latência e alta taxa de transferência.
*   **Setores de Adoção:** Telecomunicações (Verizon), Transporte (Delta), Saúde, Finanças (Goldman Sachs).

## Integration

A integração primária é feita através do **Kubeflow Pipelines SDK (KFP SDK)**, que permite definir fluxos de trabalho de ML em Python. O pipeline é então compilado para um arquivo YAML que é implantado no cluster Kubeflow.

**Exemplo de Código (Definindo um Componente de Pipeline Simples):**

```python
from kfp.v2.dsl import component
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler

# 1. Definir um componente de pipeline
@component(
    packages_to_install=['pandas'],
    base_image='python:3.9'
)
def load_data(data_path: str) -> str:
    """Carrega dados de um caminho e retorna um resumo."""
    import pandas as pd
    df = pd.read_csv(data_path)
    summary = f"Dados carregados com {len(df)} linhas e {len(df.columns)} colunas."
    print(summary)
    return summary

# 2. Definir o pipeline completo
@dsl.pipeline(
    name='simple-kubeflow-pipeline',
    description='Um pipeline simples para carregar dados.'
)
def data_pipeline(data_file: str = 'gs://my-bucket/data.csv'):
    # Usar o componente definido
    load_data_task = load_data(data_path=data_file)
    
# 3. Compilar o pipeline para um arquivo YAML (para implantação no Kubeflow)
# Compiler().compile(pipeline_func=data_pipeline, package_path='data_pipeline.yaml')
```

## URL

https://www.kubeflow.org/