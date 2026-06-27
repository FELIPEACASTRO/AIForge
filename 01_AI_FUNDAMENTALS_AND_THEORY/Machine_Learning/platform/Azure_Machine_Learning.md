# Azure Machine Learning

## Description

**Azure Machine Learning** é um serviço de nuvem de nível empresarial projetado para acelerar e gerenciar o ciclo de vida completo de projetos de Machine Learning (ML), desde o treinamento até a implantação e o gerenciamento de Operações de Machine Learning (MLOps). Sua proposta de valor única reside na sua **prontidão empresarial e segurança** (integração com Azure Virtual Networks, Key Vault e Container Registry), além de ser uma plataforma **aberta e interoperável** que suporta as principais estruturas de código aberto (PyTorch, TensorFlow, scikit-learn) e o ecossistema Azure mais amplo (Synapse Analytics, Azure Arc, Purview). A plataforma oferece ferramentas para todos os membros da equipe de ML, desde cientistas de dados que usam o Python SDK v2 até engenheiros que implementam pipelines de MLOps.

## Statistics

*   **Market Share:** Detém aproximadamente 1.34% do mercado de plataformas de Data Science e Machine Learning, competindo com mais de 150 ferramentas especializadas.
*   **Ecossistema Azure:** O Azure, como um todo, detém cerca de 20-24% do mercado global de infraestrutura em nuvem, sendo a segunda maior provedora.
*   **Adoção Empresarial:** Estima-se que 85-95% das empresas da Fortune 500 utilizam serviços do Azure, indicando uma alta adoção da plataforma em ambientes corporativos.
*   **Escalabilidade:** Suporta treinamento distribuído em múltiplos nós e GPUs de última geração, permitindo o dimensionamento de projetos de ML para qualquer tamanho necessário.

## Features

*   **MLOps e Governança:** Ferramentas robustas para gerenciar o ciclo de vida do modelo, incluindo integração com Git e MLflow, agendamento de pipeline e integração com Azure Event Grid.
*   **Suporte a LLMs e IA Generativa:** Inclui um **Catálogo de Modelos** com centenas de modelos (Azure OpenAI, Mistral, Meta, Cohere, etc.) e o **Prompt Flow** para simplificar o desenvolvimento, experimentação e implantação de aplicações de IA Generativa.
*   **Automação e Otimização:** **Automated ML (AutoML)** para seleção automatizada de recursos e algoritmos, e otimização de hiperparâmetros.
*   **Treinamento Distribuído:** Suporte para treinamento distribuído em múltiplos nós (PyTorch, TensorFlow, MPI) em clusters de computação e computação sem servidor.
*   **Implantação Gerenciada:** **Managed Endpoints** para inferência em tempo real (online) e em lote (batch), com recursos como divisão de tráfego para testes A/B.
*   **Ambiente de Desenvolvimento:** **Azure Machine Learning Studio** (UI), **Python SDK v2**, **Azure CLI v2** e **REST APIs** para diferentes perfis de usuário.

## Use Cases

*   **MLOps Empresarial:** Gerenciamento do ciclo de vida de ML de ponta a ponta em um ambiente seguro e auditável, garantindo a reprodutibilidade e a conformidade.
*   **IA Generativa e LLMs:** Construção e implantação de aplicações de IA Generativa usando modelos do Catálogo de Modelos e o Prompt Flow, como chatbots avançados e sistemas de resumo de documentos.
*   **Manutenção Preditiva:** Previsão de falhas de equipamentos com base na análise de dados de sensores, otimizando a programação de manutenção.
*   **Varejo e E-commerce:** Previsão de demanda, personalização de ofertas para clientes e gerenciamento dinâmico de estoque.
*   **Finanças:** Detecção de fraudes em tempo real e otimização de estratégias de negociação e risco.
*   **Saúde:** Desenvolvimento de modelos de diagnóstico e prognóstico com foco em segurança e conformidade regulatória.

## Integration

A integração primária é feita através do **Azure Machine Learning Python SDK v2**, que permite a criação, submissão e gerenciamento de trabalhos (jobs) de ML de forma programática.

**Exemplo de Integração (Python SDK v2 - Criação de Job):**

```python
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

# 1. Configurar o cliente ML
ml_client = MLClient(
    DefaultAzureCredential(), 
    subscription_id="<SUA_SUBSCRIPTION_ID>",
    resource_group_name="<SEU_RESOURCE_GROUP>",
    workspace_name="<SEU_WORKSPACE_NAME>"
)

# 2. Definir o comando de treinamento (Job)
job = command(
    code="./src",  # Pasta contendo o script de treinamento (ex: train.py)
    command="python train.py --data ${{inputs.input_data}}",
    inputs={
        "input_data": {
            "type": "uri_folder",
            "path": "azureml:diabetes-data:1" # Exemplo de um ativo de dados registrado
        }
    },
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest", # Ambiente Curado
    compute="<SEU_COMPUTE_CLUSTER_NAME>", # Nome do seu cluster de computação
    display_name="sklearn-training-job",
    description="Treinamento de modelo Sklearn para diabetes"
)

# 3. Submeter o trabalho
returned_job = ml_client.jobs.create_or_update(job)
print(f"Trabalho submetido. Link do Estúdio Azure ML: {returned_job.studio_url}")
```

**Outras Integrações:**
*   **MLflow:** Integração nativa para rastreamento de experimentos e registro de modelos.
*   **Azure Services:** Integração profunda com Azure Synapse Analytics (processamento de dados Spark), Azure Arc (Kubernetes), Azure Key Vault (segurança) e Azure Purview (catálogo de dados).
*   **CI/CD:** Facilidade de uso com ferramentas de CI/CD como GitHub Actions ou Azure DevOps para automação de MLOps.

## URL

https://azure.microsoft.com/en-us/products/machine-learning