# Apache Airflow

## Description

Apache Airflow é uma plataforma de código aberto, criada pela comunidade, para programar, agendar e monitorar fluxos de trabalho (workflows). Sua proposta de valor única no contexto de Machine Learning Operations (MLOps) é atuar como o coração da pilha MLOps moderna, fornecendo orquestração agnóstica a ferramentas para todo o ciclo de vida do aprendizado de máquina, desde a ingestão de dados e treinamento de modelos até a implantação e monitoramento em produção. A definição de pipelines em Python permite a integração nativa com as ferramentas de ML mais populares e a incorporação em fluxos de trabalho CI/CD de melhores práticas.

## Statistics

A adoção do Airflow em MLOps está em crescimento significativo. De acordo com o relatório 'State of Airflow 2025', 21.4% dos usuários do Airflow já o utilizam para MLOps, e 8.2% o estão adotando para fluxos de trabalho de GenAI/LLMOps. O Airflow possui uma grande comunidade, com mais de 25.000 usuários registrados no Slack e mais de 5.000 respondentes na pesquisa de 2025, demonstrando sua ampla adoção no mercado de automação de fluxo de trabalho.

## Features

Python nativo (definição de pipelines em Python, fácil integração com ferramentas de ML e TaskFlow API para tarefas Pythonic); Extensível (escrito em Python, suporta módulos e plugins customizados); Agnóstico a Dados (orquestra qualquer pipeline, independentemente do formato ou solução de armazenamento); Monitoramento e Alerta (módulos prontos para produção, como notifiers, logs extensivos e listeners); Recursos de Operação (retentativas automáticas, dependências complexas, lógica de ramificação e pipelines dinâmicos); Provedores (extensões para simplificar a integração com ferramentas populares de MLOps e serviços de nuvem como AWS, GCP e Azure).

## Use Cases

Orquestração de pipelines de Machine Learning de ponta a ponta (treinamento, avaliação, implantação); Operações de Modelos de Linguagem Grande (LLMOps), como a criação de pipelines RAG (Retrieval-Augmented Generation) para ingestão e incorporação de dados; Pipelines de Engenharia de Dados (ETL/ELT); Gerenciamento de infraestrutura e automação de tarefas de TI.

## Integration

A integração é primariamente feita através da definição de DAGs (Directed Acyclic Graphs) em Python. O TaskFlow API (introduzido no Airflow 2.0) simplifica a criação de tarefas Pythonic, permitindo que funções Python sejam transformadas em tarefas com o decorador `@task`, e o Airflow gerencia automaticamente a passagem de dados (XComs) entre elas. Exemplo de código usando TaskFlow API:\n\n```python\nimport json\nfrom airflow.decorators import dag, task\nfrom datetime import datetime\n\n@task()\ndef extract():\n    # Simula a extração de dados, como features para um modelo de ML\n    data_string = '{\"feature_a\": 10.5, \"feature_b\": 20.3}'\n    return json.loads(data_string)\n\n@task(multiple_outputs=True)\ndef transform(data_dict: dict):\n    # Simula a transformação de dados, como normalização ou engenharia de features\n    processed_value = data_dict[\"feature_a\"] * data_dict[\"feature_b\"]\n    return {\"processed_value\": processed_value}\n\n@task()\ndef load(value: float):\n    # Simula o carregamento dos dados processados, como para um modelo de inferência\n    print(f\"Valor processado carregado: {value:.2f}\")\n\n@dag(schedule=None, start_date=datetime(2023, 1, 1), catchup=False, tags=['mlops'])\ndef ml_pipeline_dag():\n    raw_data = extract()\n    transformed_data = transform(raw_data)\n    load(transformed_data[\"processed_value\"])\n\nml_pipeline_dag()\n```

## URL

https://airflow.apache.org/