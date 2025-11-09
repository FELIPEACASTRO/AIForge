# TrialBench: Multi-Modal AI-Ready Datasets for Clinical Trial Prediction

## Description

**TrialBench** é um conjunto abrangente de 23 *datasets* prontos para IA, curados a partir de mais de 480.000 registros de ensaios clínicos do ClinicalTrials.gov (dados até fevereiro de 2024), enriquecidos com informações do DrugBank e TrialTrove. O objetivo é facilitar o desenvolvimento de modelos de Inteligência Artificial para prever eventos críticos e otimizar o design de ensaios clínicos. Os *datasets* são multimodais, incluindo dados tabulares, texto livre (critérios de elegibilidade) e dados de grafos (estrutura molecular de medicamentos), e são organizados em torno de 8 tarefas de previsão cruciais. Foi publicado em 2025, garantindo sua relevância e atualidade.

## Statistics

O TrialBench é composto por 23 *datasets* derivados de mais de **480.000** registros de ensaios clínicos (até fev/2024). A tabela a seguir resume o número de ensaios (em milhares) por tarefa:
| Tarefas | # Ensaios (Total) | # Ensaios (Fase I/II/III/IV) |
| :--- | :--- | :--- |
| Previsão de Duração | 143.8K | 13.5K/13.4K/9.2K/7.1K |
| Previsão de Abandono | 62.1K | 4.2K/15.8K/11.5K/6.9K |
| Previsão de Evento Adverso Grave | 31.3K | 2.0K/8.1K/4.8K/2.9K |
| Previsão de Mortalidade | 31.3K | 2.0K/8.1K/4.8K/2.9K |
| Previsão de Aprovação | 43.2K | 4.5K/12.5K/9.2K/4.5K |
| Identificação da Razão de Falha | 41.4K | 4.3K/8.8K/4.2K/3.5K |
| Design de Critérios de Elegibilidade | 136.4K | 19.4K/14.2K/10.8K/10.6K |
| Determinação de Dosagem | 12.8K | 0/12.8K/0/0 |
O conjunto total inclui 40.8K ensaios com medicamentos, 21.1K com dispositivos médicos e 83.6K com outras intervenções.

## Features

Conjunto de 23 *datasets* prontos para IA, cobrindo 8 tarefas de previsão: duração do ensaio, taxa de abandono do paciente, evento adverso grave, mortalidade, resultado de aprovação, razão de falha, dosagem de medicamento e design de critérios de elegibilidade. Os *datasets* são **multimodais**, incorporando: 1) **Características Categóricas e Numéricas** (ex: tipo de estudo, idade); 2) **Texto Livre** (ex: critérios de elegibilidade, resumo do ensaio); 3) **Dados de Grafos** (ex: estrutura molecular de medicamentos via SMILES); 4) **Termos MeSH** e **Códigos ICD-10** para doenças.

## Use Cases

O principal caso de uso é a **otimização do design de ensaios clínicos** através da aplicação de IA. Isso inclui:
1.  **Previsão de Riscos:** Estimar a duração do ensaio, a probabilidade de abandono do paciente, e a ocorrência de eventos adversos graves ou mortalidade.
2.  **Otimização de Resultados:** Prever o resultado de aprovação do ensaio e identificar as razões prováveis de falha.
3.  **Suporte à Decisão:** Auxiliar na determinação da dosagem ideal de medicamentos e no design de critérios de elegibilidade mais eficazes.
O recurso é ideal para pesquisadores e cientistas de dados que trabalham com *Machine Learning* e *Deep Learning* em **Informática Médica** e **IA em Saúde**.

## Integration

A integração é facilitada por pacotes dedicados em **Python** (`pip install trialbench`) e **R**. O pacote Python permite o download e carregamento dos *datasets* em formatos otimizados para *Deep Learning* (DL) ou como *DataFrames* do Pandas.
**Exemplo de Acesso (Python):**
```python
import trialbench
# Download de todos os datasets (opcional)
trialbench.function.download_all_data('data/')
# Carregar dados para uma tarefa específica (ex: dosagem)
task = 'dose'
phase = 'All'
# Formato Dataloader (para DL)
train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='dl')
# Ou como Pandas DataFrame
train_df, valid_df, test_df, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='df')
```

## URL

https://www.nature.com/articles/s41597-025-05680-8