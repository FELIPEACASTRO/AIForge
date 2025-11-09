# Evidently AI

## Description

Evidently é um framework de observabilidade de Machine Learning (ML) e Large Language Models (LLM) de código aberto, baseado em Python. Sua função principal é avaliar, testar e monitorar a qualidade e o desempenho de qualquer sistema alimentado por IA ou pipeline de dados, abrangendo desde dados tabulares tradicionais até aplicações de Geração de IA (Gen AI). A proposta de valor única reside na sua modularidade, oferecendo mais de 100 métricas integradas para detecção de desvios (drift), avaliação de modelos e testes de qualidade de dados, sendo uma ferramenta essencial para MLOps.

## Statistics

**Estatísticas do Repositório GitHub (evidentlyai/evidently):** Estrelas: 6.8k, Forks: 740, Commits: 2,715. **Métricas de Monitoramento:** Suporta mais de 100 métricas, incluindo PSI, K-L divergence para Data Drift; Acurácia, F1 Score para Classificação; MAE, RMSE para Regressão; e Sentimento, Toxicidade para LLM/Texto.

## Features

**Reports:** Calcula e resume avaliações de qualidade de dados, ML e LLM, ideal para análise exploratória e depuração. Pode ser exportado como JSON, dicionário Python, HTML ou visualizado na UI. **Test Suites:** Transforma Reports em condições de aprovação/falha (pass/fail conditions), ideal para testes de regressão, CI/CD e validação de dados. **Monitoring Dashboard (UI):** Serviço para visualizar métricas e resultados de testes ao longo do tempo, disponível para auto-hospedagem (open-source) ou via Evidently Cloud. **Métricas Extensivas:** Inclui mais de 100 métricas para Data Drift (PSI, K-L divergence), Qualidade de Dados, Classificação, Regressão, Ranking e LLM/Texto (Sentimento, Toxicidade, Relevância RAG).

## Use Cases

**Validação de Modelos:** Garantir que os modelos de ML e LLM estejam prontos para produção. **Detecção de Drift:** Identificar desvios na distribuição de dados (Data Drift) ou no desempenho do modelo (Model Drift) em produção. **Testes de Regressão:** Usar Test Suites em pipelines de CI/CD para garantir que novas versões de dados ou modelos não introduzam problemas. **Observabilidade de LLM:** Monitorar a qualidade das respostas de LLMs, incluindo sentimentos, toxicidade e relevância em sistemas RAG. **Análise Exploratória:** Usar Reports para entender a qualidade de novos conjuntos de dados ou o desempenho de modelos em experimentos.

## Integration

Evidently é uma biblioteca Python que se integra facilmente com outras ferramentas de MLOps. A instalação é feita via `pip install evidently`.

**Exemplo de Detecção de Data Drift (Python):**
```python
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn import datasets

# 1. Preparar dados de referência e atuais
iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame
reference_data = iris_frame.iloc[:60]
current_data = iris_frame.iloc[60:]

# 2. Criar e executar o Report de Data Drift
report = Report([
    DataDriftPreset(method="psi")
])
drift_report = report.run(reference_data=reference_data, current_data=current_data)

# 3. Salvar o resultado em HTML
drift_report.save_html("data_drift_report.html")
```
**Integrações com Outras Ferramentas:** MLflow, Neptune, Airflow, Kubeflow (para logar métricas e relatórios) e pipelines de CI/CD (como GitHub Actions) para validação pré-deploy.

## URL

https://www.evidentlyai.com/