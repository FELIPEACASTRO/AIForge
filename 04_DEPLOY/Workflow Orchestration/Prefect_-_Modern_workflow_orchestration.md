# Prefect - Modern workflow orchestration

## Description

O Prefect é um framework de orquestração de fluxo de trabalho moderno e de código aberto, projetado para transformar funções Python em pipelines de dados de nível de produção com atrito mínimo. Sua proposta de valor única reside em sua abordagem "Pythonic" e dinâmica, que permite aos engenheiros de dados e desenvolvedores construir, observar e reagir a fluxos de trabalho complexos de forma resiliente e escalável. Ao contrário dos orquestradores tradicionais baseados em DAGs estáticos, o Prefect utiliza um modelo de fluxo de trabalho dinâmico que lida nativamente com falhas, reexecuções e lógica condicional complexa, tornando-o ideal para pipelines de Machine Learning e Data Science. Ele se posiciona como uma solução de orquestração para a era moderna da engenharia de dados, focada em observabilidade e experiência do desenvolvedor.

## Statistics

**Estrelas no GitHub:** Aproximadamente 18.5K estrelas (referência de 2024/2025). **Downloads:** Mais de 7.2 milhões de downloads (referência de 2024/2025, versão 3.0). **Adoção:** Adotado por empresas como Snorkel AI para executar milhares de fluxos de trabalho diariamente. **Comunidade:** Embora menor que a do Apache Airflow, está em rápido crescimento, com foco em uma comunidade de desenvolvedores Python e engenheiros de dados que buscam soluções mais modernas e dinâmicas. **Categoria:** Ferramenta de Orquestração de Fluxo de Trabalho (Workflow Orchestration Tool).

## Features

**Orquestração Pythonic e Dinâmica:** Qualquer função Python pode se tornar um fluxo de trabalho (Flow) com o decorador `@flow`, permitindo a construção de DAGs dinâmicos em tempo de execução. **Observabilidade Integrada:** Oferece um painel de controle (Prefect UI) para monitoramento em tempo real, logs, métricas e rastreamento de estado de ponta a ponta. **Resiliência e Tratamento de Falhas:** Inclui recursos nativos para novas tentativas (retries), cache e lógica de estado, garantindo que os fluxos de trabalho sejam robustos contra falhas. **Blocos (Blocks):** Componentes configuráveis e reutilizáveis para interagir com sistemas externos (ex: AWS S3, Google Cloud Storage, dbt Cloud) sem expor credenciais no código. **Implantações (Deployments):** Mecanismo para empacotar, agendar e executar fluxos de trabalho de forma consistente em qualquer ambiente (local, Docker, Kubernetes). **Integrações Extensivas:** Suporte robusto para ecossistemas de dados populares como dbt, Spark, Pandas, e ferramentas de ML/AI.

## Use Cases

**Pipelines de Engenharia de Dados:** Orquestração de ETL/ELT complexos, garantindo resiliência e observabilidade em todas as etapas de extração, transformação e carregamento de dados. **Fluxos de Trabalho de Machine Learning (MLOps):** Gerenciamento de pipelines de ML, incluindo treinamento de modelos, validação, registro e implantação, com a capacidade de lidar com a natureza dinâmica e condicional desses fluxos. **Processamento de Eventos em Tempo Real:** Uso de webhooks e automações para acionar fluxos de trabalho instantaneamente em resposta a eventos externos (ex: mudanças no banco de dados via Debezium, upload de arquivos). **Automação de Tarefas de TI e Negócios:** Agendamento e monitoramento de tarefas recorrentes, como geração de relatórios, backups de banco de dados e sincronização de sistemas. **Orquestração de dbt:** Integração nativa para executar e monitorar projetos dbt, adicionando resiliência e observabilidade aos modelos de transformação de dados.

## Integration

A integração com o Prefect é centrada no Python, utilizando decoradores para definir fluxos de trabalho e a CLI ou SDK para implantação.

**1. Definição de Fluxo de Trabalho (Flow) Básico:**
```python
from prefect import flow, task

@task
def extrair_dados():
    print("Extraindo dados...")
    return [1, 2, 3]

@task
def transformar_dados(data):
    print(f"Transformando {data}...")
    return [x * 2 for x in data]

@flow(name="Meu Primeiro Flow Prefect")
def pipeline_principal():
    dados_brutos = extrair_dados()
    dados_transformados = transformar_dados(dados_brutos)
    print(f"Resultado final: {dados_transformados}")

if __name__ == "__main__":
    pipeline_principal()
```

**2. Integração com dbt (Exemplo usando `prefect-dbt`):**
Para orquestrar um projeto dbt, utiliza-se a biblioteca de integração `prefect-dbt`.

```python
from prefect import flow
from prefect_dbt.cli import DbtCliProfile, DbtCoreOperation

@flow(name="dbt Orchestration Flow")
def dbt_flow_run():
    # O bloco DbtCliProfile armazena as configurações de conexão do dbt
    dbt_profile = DbtCliProfile.load("my-dbt-profile") 
    
    # Executa o comando 'dbt run'
    DbtCoreOperation(
        commands=["dbt run"],
        dbt_cli_profile=dbt_profile,
        project_dir="/caminho/para/seu/projeto/dbt"
    ).run()

if __name__ == "__main__":
    dbt_flow_run()
```

**3. Implantação (Deployment) via Python:**
O Prefect usa o conceito de `Deployment` para agendar e executar fluxos de trabalho em ambientes de produção.

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
from datetime import timedelta

# Assumindo que 'pipeline_principal' está definido no arquivo 'meu_flow.py'
deployment = Deployment.build_from_flow(
    flow=pipeline_principal,
    name="meu-pipeline-diario",
    version="1.0",
    schedule=IntervalSchedule(interval=timedelta(days=1)),
    work_pool_name="default-worker-pool"
)

if __name__ == "__main__":
    deployment.apply() # Aplica a implantação ao Prefect Server
```

## URL

https://www.prefect.io/ | https://github.com/PrefectHQ/prefect