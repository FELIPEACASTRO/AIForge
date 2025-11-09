# TransformEHR

## Description

Modelo generativo de codificador-decodificador baseado em Transformer, pré-treinado com um novo objetivo de prever todas as doenças e resultados de um paciente em uma visita futura a partir de visitas anteriores. Projetado para melhorar a previsão de resultados de doenças usando Prontuários Eletrônicos de Saúde (PES) longitudinais. Demonstrou ser capaz de capturar relações complexas em dados longitudinais de PES.

## Statistics

Superou modelos anteriores (como BERT, LSTM) em tarefas de previsão.
*   **Câncer de Pâncreas:** AUROC de 81,95%.
*   **Autoagressão Intencional (em pacientes com PTSD):** AUPRC melhorada em 24% em relação ao BERT (AUPRC de 16,67%).
*   **Citações:** 106 (em 2023).

## Features

Arquitetura Transformer encoder-decoder; Novo objetivo de pré-treinamento para previsão de doenças futuras; Capacidade de fine-tuning com dados limitados; Alto desempenho em tarefas de previsão clínica.

## Use Cases

Previsão de doenças raras (ex: câncer de pâncreas); Previsão de resultados clínicos críticos (ex: autoagressão intencional em pacientes com PTSD); Transferência de aprendizado para novos conjuntos de dados de PES.

## Integration

O código de fine-tuning do TransformEHR está disponível publicamente no GitHub. O modelo pode ser adaptado para diversas tarefas de previsão clínica.
**Exemplo de Repositório:**
```
https://github.com/whaleloops/TransformEHR/
```

## URL

https://www.nature.com/articles/s41467-023-43715-z