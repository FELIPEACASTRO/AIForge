# Model Registry Solutions - MLflow Registry, Seldon, BentoML

## Description

O MLflow Model Registry é um repositório centralizado, com APIs e interface de usuário (UI), projetado para gerenciar colaborativamente o ciclo de vida completo de um modelo de Machine Learning (ML). Sua proposta de valor única reside na sua integração nativa com o ecossistema MLflow mais amplo (Tracking, Projects e Serving), fornecendo uma solução MLOps de código aberto e padronizada. Ele permite o versionamento de modelos, a transição de estágios (Staging, Production, Archived) e a rastreabilidade completa desde o código de treinamento até a implantação em produção. O Seldon Core é uma plataforma de código aberto para implantação de modelos de ML em ambientes de produção, com foco em Kubernetes. Sua proposta de valor é a capacidade de construir e gerenciar grafos de inferência complexos (Inference Graphs), permitindo a composição de modelos, transformadores de dados, roteadores e testes A/B. É ideal para cenários que exigem alta escalabilidade, baixa latência e arquiteturas de microsserviços. O BentoML é uma estrutura unificada de inferência de IA que transforma modelos de ML treinados em serviços de produção prontos para implantação. Sua proposta de valor única é a criação de "Bentos" (pacotes de modelo e código de serviço) que são facilmente implantáveis em qualquer ambiente de nuvem ou infraestrutura. Ele simplifica o processo de empacotamento, otimização e escalabilidade de modelos, suportando uma ampla gama de frameworks de ML e focando na eficiência de inferência.

## Statistics

**MLflow:** Mais de 16 milhões de downloads mensais (2024), amplamente adotado por empresas como Databricks, DoorDash e Walmart. **Seldon:** Utilizado por centenas de empresas, com um aumento de 5x nos clientes empresariais desde 2021 (Seldon Deploy). Focado em implantações de baixa latência (p99 < 25ms em cenários de produção). **BentoML:** Mais de 336 empresas utilizam, com uma comunidade ativa e crescimento de adoção de ferramentas de código aberto de IA em 30% (2024). A empresa por trás levantou $9M em financiamento.

## Features

**MLflow Model Registry:** Versionamento de modelos, transição de estágios (Staging, Production, Archived), anotações e descrições de modelos, rastreabilidade de linhagem, integração com CI/CD. **Seldon Core:** Implantação em Kubernetes, grafos de inferência complexos (roteamento, testes A/B, canários), suporte a múltiplos frameworks (TensorFlow, PyTorch, Scikit-learn), monitoramento e métricas operacionais (Prometheus/Grafana). **BentoML:** Criação de "Bentos" (pacotes de modelo e serviço), otimização de inferência (batching adaptativo, quantização), suporte a LLMs, API de serviço unificada (REST/gRPC), escalabilidade automática e implantação em diversas plataformas (Docker, Kubernetes, Cloud).

## Use Cases

**MLflow Model Registry:** Gerenciamento de modelos em ambientes regulamentados, rastreabilidade de modelos para auditoria, transição automatizada de modelos para produção via CI/CD. **Seldon Core:** Implantação de sistemas de recomendação em tempo real, detecção de fraude, roteamento de tráfego para testes A/B de modelos, composição de pipelines de inferência complexos (pré-processamento + modelo + pós-processamento). **BentoML:** Servir modelos de Linguagem Grande (LLMs) com otimização de inferência, implantação de APIs de predição de alta performance para aplicações web e móveis, empacotamento de modelos para edge computing.

## Integration

**MLflow Model Registry:** Integração via Python Client: `client.transition_model_version_stage(name="model_name", version=1, stage="Production")`. Registro de modelo: `mlflow.register_model("runs:/run_id/model", "model_name")`. **Seldon Core:** Implantação via Custom Resource Definition (CRD) do Kubernetes. Exemplo de YAML para implantação de um modelo:
```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
spec:
  name: iris-deployment
  predictors:
  - graph:
      children: []
      model:
        name: classifier
        uri: gs://seldon-models/sklearn/iris
        type: MODEL
    name: default
    replicas: 1
```
**BentoML:** Empacotamento e execução de serviço via Python CLI. Exemplo de serviço:
```python
import bentoml
from bentoml.io import JSON
# Define o serviço com um executor de modelo
svc = bentoml.Service("iris_classifier", runners=[iris_runner])
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # Lógica de pré-processamento e chamada ao modelo
    result = iris_runner.predict.run(input_data)
    return {"prediction": result.tolist()}
# Cria o Bento: bentoml build
```

## URL

**MLflow Model Registry:** https://mlflow.org/docs/latest/ml/model-registry/ **Seldon Core:** https://www.seldon.io/ **BentoML:** https://www.bentoml.com/