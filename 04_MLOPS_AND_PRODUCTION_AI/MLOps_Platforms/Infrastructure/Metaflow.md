# Metaflow

## Description

Metaflow é um framework de código aberto para infraestrutura de Machine Learning (ML) e Ciência de Dados, originalmente desenvolvido na Netflix. Ele se destaca por ser uma **biblioteca Python amigável** que oferece uma API unificada para todo o *stack* de infraestrutura necessário para desenvolver, implantar e operar aplicações intensivas em dados, desde o protótipo até a produção. Sua proposta de valor única reside em permitir que cientistas de dados e engenheiros transitem de forma fluida entre o desenvolvimento local (em notebooks ou *laptops*) e a execução em escala na nuvem (AWS, Azure, GCP, Kubernetes), com **versionamento automático** e **alta disponibilidade** em produção, sem a necessidade de reescrever o código. O Metaflow é projetado para simplificar a complexidade do MLOps, focando na produtividade e na capacidade de testar e validar projetos em escala antes de um investimento total em infraestrutura.

## Statistics

*   **Origem**: Desenvolvido e utilizado internamente na Netflix para gerenciar milhares de fluxos de ML e milhões de execuções.
*   **Licença**: Código aberto sob a Licença Apache 2.0.
*   **Adoção**: Utilizado por empresas como Outerbounds (a empresa por trás do Metaflow), CNN, e outras que buscam padronizar seus fluxos de trabalho de ML.
*   **Escalabilidade Comprovada**: Projetado para lidar com dezenas de milhares de fluxos e milhões de execuções em produção, demonstrando robustez e capacidade de escala.

## Features

*   **API Unificada**: Uma única biblioteca Python para gerenciar modelagem, *deployment*, versionamento, orquestração, computação e dados.
*   **Transição Suave**: Permite o desenvolvimento local e a execução em escala na nuvem (AWS, Azure, GCP, Kubernetes) sem alterações no código.
*   **Versionamento Automático**: Rastreia automaticamente todos os fluxos, experimentos e artefatos de dados.
*   **Escalabilidade na Nuvem**: Integração nativa com AWS Batch, AWS Step Functions, Kubernetes, Argo Workflows e Apache Airflow para computação e agendamento em larga escala.
*   **Isolamento de Dependências**: Usa `conda` ou `docker` para garantir que as dependências do código sejam reproduzíveis em todos os ambientes.
*   **Visualização de Resultados**: Mecanismo integrado de *Cards* para criar e visualizar relatórios com imagens e texto.

## Use Cases

*   **Treinamento de Modelos em Larga Escala**: Executar o treinamento de modelos de *Deep Learning* e *Machine Learning* que exigem grandes volumes de dados e recursos computacionais, aproveitando a integração com Kubernetes e AWS Batch.
*   **Pipelines de Dados e ML Reproduzíveis**: Criar fluxos de trabalho que garantem a rastreabilidade e a reprodutibilidade de experimentos, desde a ingestão de dados até a *deployment* do modelo, graças ao versionamento automático de artefatos.
*   **Análise Estatística e Relatórios**: Utilizado para carregar metadados, calcular estatísticas específicas de domínio (como estatísticas de gênero de filmes na Netflix) e gerar relatórios visuais (Cards) para *stakeholders*.
*   **Desenvolvimento Iterativo de ML**: Suportar a jornada do protótipo à produção, permitindo que cientistas de dados desenvolvam rapidamente em um *notebook* e, em seguida, escalem para a nuvem com um comando simples (`--run-id`).
*   **Sistemas de Recomendação**: Na Netflix, o Metaflow é usado para construir e gerenciar os complexos *pipelines* de dados e modelos que alimentam seus sistemas de recomendação.

## Integration

O Metaflow se integra perfeitamente com os principais serviços de nuvem e orquestradores de fluxo de trabalho. A integração é tipicamente feita através de **decoradores Python** que abstraem a complexidade da infraestrutura.

**Exemplo de Integração com Kubernetes para Escala:**

```python
from metaflow import FlowSpec, step, kubernetes

class ScalingFlow(FlowSpec):
    @kubernetes(memory=64000, cpu=16) # Solicita 64GB de RAM e 16 vCPUs no Kubernetes
    @step
    def start(self):
        # Lógica de processamento intensivo que será executada em um pod K8s
        self.data = self.process_large_dataset()
        self.next(self.end)

    @step
    def end(self):
        print("Processamento concluído com sucesso.")

if __name__ == '__main__':
    ScalingFlow()
```

**Exemplo de Integração com S3 para Armazenamento de Dados:**

O Metaflow gerencia automaticamente o armazenamento de artefatos no S3 (ou outro *backend* de nuvem) e fornece um cliente S3 amigável:

```python
from metaflow import FlowSpec, step, S3

class DataFlow(FlowSpec):
    @step
    def start(self):
        # O cliente S3 do Metaflow simplifica o acesso a dados
        with S3(run=self) as s3:
            # Baixa um arquivo do S3
            s3.get("s3://my-bucket/input.csv", "local_input.csv")
            # Faz o upload de um artefato
            s3.put("s3://my-bucket/output.txt", "conteúdo do arquivo")
        self.next(self.end)

    @step
    def end(self):
        pass
```

**Agendamento com Airflow:**

Para agendar um fluxo Metaflow com Apache Airflow, basta gerar o DAG correspondente a partir do fluxo Python:

```bash
python my_metaflow_flow.py --datastore=s3 airflow create
```

Este comando gera um arquivo `my_metaflow_flow_airflow_dag.py` que pode ser implantado no Airflow.

## URL

https://metaflow.org/