# Data Quality Tools: Great Expectations, Deequ, Soda

## Description

**Great Expectations (GX)** é a principal ferramenta de código aberto para validação e documentação de dados, atuando como testes de unidade para dados. Sua proposta de valor única é fornecer uma linguagem comum para qualidade de dados por meio de "Expectations" (Expectativas), que são asserções de dados expressivas e extensíveis. O GX ajuda as equipes a validar dados críticos em seus pipelines, construir confiança nos dados e gerar automaticamente a "Data Docs" (Documentação de Dados), um site de documentação de dados pesquisável e visualmente rico.
\n\n**Deequ** é uma biblioteca de código aberto construída sobre o Apache Spark, desenvolvida pela Amazon, para definir testes de unidade para dados e medir a qualidade dos dados em grandes conjuntos de dados em escala. Sua proposta de valor reside em sua capacidade de calcular métricas de qualidade de dados, definir e verificar restrições de qualidade de dados e ser informada sobre alterações nos dados, tudo isso aproveitando o poder de processamento distribuído do Spark. É ideal para pipelines de dados baseados em Spark que exigem validação de qualidade de dados em tempo de execução.
\n\n**Soda** é uma plataforma de qualidade de dados nativa de IA que fornece ferramentas para monitorar, testar e melhorar a qualidade dos dados em todas as pilhas. O **Soda Core** é a parte de código aberto, uma biblioteca Python e ferramenta CLI que permite aos engenheiros de dados usar a "Soda Checks Language" (Linguagem de Verificação Soda) para transformar entradas definidas pelo usuário em consultas SQL para testes de qualidade de dados. Sua proposta de valor única é a "Metrics Observability" (Observabilidade de Métricas) que detecta anomalias de forma mais rápida e precisa do que os sistemas tradicionais, permitindo que as equipes encontrem, compreendam e corrijam problemas de qualidade de dados em segundos.

## Statistics

**Great Expectations:**
\n- **GitHub Stars:** ~10.5k+ (great-expectations/great_expectations)
\n- **PyPI Downloads:** Milhões de downloads mensais.
\n- **Adoção:** Amplamente adotado em ambientes Python e em pipelines de dados modernos.
\n\n**Deequ:**
\n- **GitHub Stars:** ~4.5k+ (awslabs/deequ)
\n- **Tecnologia:** Baseado em Apache Spark, o que o torna altamente escalável para Big Data.
\n- **Origem:** Desenvolvido e mantido pela Amazon Web Services (AWS).
\n\n**Soda:**
\n- **Soda Core GitHub Stars:** ~1.5k+ (sodadata/soda-core)
\n- **Modelo:** Oferece uma versão de código aberto (Soda Core) e uma plataforma comercial (Soda Cloud).
\n- **Diferencial:** Foco em Observabilidade de Dados e detecção de anomalias nativa de IA.

## Features

**Great Expectations:**
\n- **Expectations:** Asserções de dados expressivas e extensíveis (mais de 50 Expectativas prontas para uso).
\n- **Data Docs:** Geração automática de documentação de dados pesquisável e visualmente rica.
\n- **Data Context:** Gerenciamento de configurações, Expectations e resultados.
\n- **Validation:** Execução de Expectations em conjuntos de dados para produzir resultados de validação.
\n\n**Deequ:**
\n- **Constraint Checking:** Definição e verificação de restrições de qualidade de dados (completude, unicidade, validade, etc.).
\n- **Metric Computation:** Cálculo de métricas de qualidade de dados (contagem, média, desvio padrão, etc.) em escala usando Spark.
\n- **Anomaly Detection:** Detecção de anomalias em métricas de qualidade de dados ao longo do tempo.
\n- **Profiling:** Geração de perfis de dados para descobrir automaticamente restrições de qualidade de dados.
\n\n**Soda:**
\n- **Soda Checks Language (SCL):** Linguagem YAML para definir verificações de qualidade de dados.
\n- **Metrics Observability:** Detecção de anomalias nativa de IA e monitoramento contínuo.
\n- **Data Source Connectors:** Suporte para mais de 20 fontes de dados (Snowflake, Databricks, BigQuery, etc.).
\n- **Scan Execution:** Execução de verificações de qualidade de dados em qualquer estágio do pipeline.

## Use Cases

**Great Expectations:**
\n- **Validação de Ingestão de Dados:** Garantir que os dados de entrada atendam aos padrões de qualidade antes do processamento.
\n- **Monitoramento de Pipeline:** Adicionar Expectations em pontos críticos do pipeline (staging, transformação, saída) para evitar a propagação de dados ruins.
\n- **Documentação de Dados:** Criar um catálogo de dados vivo e pesquisável (Data Docs) para comunicação entre equipes.
\n\n**Deequ:**
\n- **Testes de Unidade para Dados em Escala:** Executar verificações de qualidade de dados em grandes conjuntos de dados no Spark.
\n- **Monitoramento de Qualidade em ETL/ELT:** Integrar verificações de qualidade diretamente em trabalhos Spark para falhar rapidamente em caso de violações.
\n- **Detecção de Desvio de Dados:** Monitorar métricas de dados ao longo do tempo para identificar desvios e anomalias.
\n\n**Soda:**
\n- **Observabilidade Contínua de Dados:** Monitorar a qualidade dos dados em tempo real para detectar anomalias e problemas de frescor.
\n- **Qualidade de Dados em Data Mesh:** Permitir que as equipes de domínio definam e gerenciem suas próprias verificações de qualidade de dados (Data as a Product).
\n- **Testes de Qualidade de Dados em CI/CD:** Integrar verificações de qualidade em fluxos de trabalho de CI/CD (como GitHub Actions) para evitar a implantação de código que introduza dados ruins.

## Integration

**Great Expectations (Python):**
\nIntegra-se com Python, Spark, Pandas e várias fontes de dados (SQL, S3, GCS).
\n```python
\nimport great_expectations as gx
\ncontext = gx.get_context()
\nbatch_request = context.data_sources.all()[0].build_batch_request(data_asset_name="my_table")
\ncheckpoint = context.add_or_update_checkpoint(
\n    name="my_checkpoint",
\n    batch_request=batch_request,
\n    expectation_suite_name="my_suite",
\n)
\ncheckpoint_result = checkpoint.run()
\n```
\n\n**Deequ (Scala/Python):**
\nConstruído sobre o Apache Spark, usa Scala ou PyDeequ (API Python).
\n```scala
\nimport com.amazon.deequ.checks.{Check, CheckLevel}
\nimport com.amazon.deequ.VerificationSuite
\n\nval verificationResult = VerificationSuite()
\n  .onData(data)
\n  .addCheck(
\n    Check(CheckLevel.Error, "unit testing data")
\n      .hasSize(_ >= 5)
\n      .isComplete("id")
\n      .isUnique("id")
\n  ).run()
\n```
\n\n**Soda (CLI/YAML):**
\nUsa a ferramenta CLI e arquivos YAML para definir verificações.
\n```bash
\n# Instalação
\npip install soda-core-snowflake
\n# Execução de verificação
\nsoda scan -d snowflake -c configuration.yml checks.yml
\n```
\n*checks.yml:*
\n```yaml
\nchecks for dim_customer:
\n  - row_count > 0
\n  - missing_count(customer_id) = 0
\n  - invalid_count(email) = 0:
\n      valid format: email
\n```

## URL

Great Expectations: https://greatexpectations.io/\nDeequ: https://github.com/awslabs/deequ\nSoda: https://www.soda.io/