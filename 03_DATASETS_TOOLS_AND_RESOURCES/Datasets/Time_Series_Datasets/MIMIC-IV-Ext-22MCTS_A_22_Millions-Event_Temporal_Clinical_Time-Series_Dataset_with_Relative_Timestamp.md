# MIMIC-IV-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp

## Description

O MIMIC-IV-Ext-22MCTS é um conjunto de dados de eventos clínicos temporais em série temporal, com informações temporais concretas. É derivado do **MIMIC-IV-Note**, que contém notas clínicas desidentificadas. Este dataset foi criado para facilitar a modelagem de eventos clínicos temporais, extraindo eventos clínicos como pequenos trechos de texto e seus respectivos *timestamps* relativos a partir de sumários de alta, utilizando técnicas de recuperação contextual e o modelo Llama-3.1-8B. O *timestamp* é dado em horas, sendo negativo para eventos históricos (antes da admissão) e positivo para eventos durante ou após a admissão. O dataset também inclui uma coluna `Time_bin` que discretiza o tempo em 9 categorias predefinidas. É um recurso valioso para pré-treinamento e tarefas de aprendizado semi-supervisionado ou com supervisão fraca.

## Statistics

**Total de Sumários de Alta:** 267.284
**Total de Registros (Pares Evento-Timestamp):** 22.588.586
**Eventos por Sumário (Mín/Máx):** 1 / 244
**Eventos por Sumário (Média):** 84
**Distribuição Temporal dos Eventos:**
- Antes da admissão (Histórico): 36.99%
- Durante a admissão: 51.19%
- Após a alta (Futuro): 11.80%
**Tokens por Evento (Média):** 3
**Tokens por Evento (Máx):** 299
**Publicação:** Setembro de 2025 (Versão 1.0.0)

## Features

**Dados de Eventos Clínicos Temporais:** Consiste em eventos clínicos extraídos de sumários de alta, cada um associado a um *timestamp* relativo.
**Timestamp Relativo:** O tempo é medido em horas, relativo ao momento da admissão.
**Discretização Temporal (`Time_bin`):** O tempo contínuo é mapeado em 9 categorias discretas (Bins), facilitando a modelagem temporal.
**Fonte:** Derivado do dataset MIMIC-IV-Note, garantindo a base em dados clínicos reais e desidentificados.
**Aplicações de *Fine-tuning*:** Utilizado para *fine-tuning* de modelos como BERT e GPT-2 para tarefas de P&R e correspondência de ensaios clínicos.

## Use Cases

**Pré-treinamento de Modelos de Linguagem (LLMs):** Ideal para pré-treinar modelos como GPT-2 para gerar saídas mais orientadas clinicamente.
**Modelagem de Eventos Clínicos Temporais:** Utilizado para desenvolver e testar modelos que preveem a sequência e o tempo de eventos clínicos.
**Correspondência de Ensaios Clínicos:** *Fine-tuning* de modelos (ex: BERT) para melhorar a correspondência de pacientes com critérios de ensaios clínicos.
**Previsão de Risco Clínico:** Embora com limitações para avaliação de alto risco devido à falta de rótulos de verdade fundamental (*Ground Truth*), é útil para tarefas de modelagem preditiva e aprendizado semi-supervisionado.

## Integration

O dataset está disponível no PhysioNet e requer **acesso credenciado** (Credentialed Access) para download, seguindo as políticas de uso do PhysioNet.
O acesso geralmente envolve a conclusão de um curso de treinamento em proteção de dados humanos e a assinatura de um Acordo de Uso de Dados (DUA).
**Estrutura da Tabela:**
- `Hadm_id`: Identificador único para cada sumário de alta.
- `Event`: O evento clínico em formato de texto.
- `Time`: O *timestamp* do evento em horas (contínuo).
- `Time_bin`: A categoria discreta do *timestamp* (0 a 8).
O código relacionado para *fine-tuning* de modelos (como BERT) para explorar a relação causal entre eventos clínicos está **disponível publicamente** (embora o URL específico não tenha sido fornecido na página de resumo).

## URL

https://physionet.org/content/mimic-iv-ext-22mcts/