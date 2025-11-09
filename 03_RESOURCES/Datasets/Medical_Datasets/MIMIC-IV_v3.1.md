# MIMIC-IV v3.1

## Description

O Medical Information Mart for Intensive Care IV (MIMIC-IV) é um banco de dados desidentificado e de acesso livre, derivado de registros eletrônicos de saúde (EHR) do Beth Israel Deaconess Medical Center. A versão 3.1 é a mais recente, publicada em 2024, e é amplamente utilizada para pesquisa em informática médica e IA, fornecendo dados ricos e complexos de pacientes de UTI e emergência. É a principal fonte de dados para o desenvolvimento de modelos de IA em cuidados críticos.

## Statistics

Pacientes Únicos: 364.627; Hospitalizações: 546.028; Estadias em UTI: 94.458 (para mais de 65.000 pacientes); Publicação: Outubro de 2024 (v3.1). Estrutura organizada em módulos `hosp` (dados hospitalares gerais) e `icu` (dados de UTI).

## Features

Inclui dados demográficos, comorbidades, diagnósticos (ICD), procedimentos, prescrições, administração de medicamentos, resultados de microbiologia, e, crucialmente para este tópico, **sinais vitais** (`chartevents`) e **resultados laboratoriais** (`labevents`). Os sinais vitais incluem frequência cardíaca, pressão arterial, saturação de oxigênio, frequência respiratória e temperatura.

## Use Cases

Previsão de mortalidade hospitalar, detecção precoce de sepse, modelagem de progressão de doenças crônicas, desenvolvimento de modelos de risco personalizados, e pesquisa sobre engenharia de features de séries temporais clínicas.

## Integration

Acesso via PhysioNet (após credenciamento e assinatura de acordo de uso de dados). Os dados são fornecidos em formato tabular (CSV) e podem ser carregados em bancos de dados como PostgreSQL ou BigQuery (MIMIC-IV v3.1 está disponível no BigQuery).

## URL

https://physionet.org/content/mimiciv/