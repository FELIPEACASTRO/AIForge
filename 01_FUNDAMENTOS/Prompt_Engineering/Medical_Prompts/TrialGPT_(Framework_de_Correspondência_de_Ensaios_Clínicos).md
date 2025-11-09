# TrialGPT (Framework de Correspondência de Ensaios Clínicos)

## Description

TrialGPT é um framework ponta a ponta, agnóstico a LLM (utilizando primariamente GPT-4 e GPT-3.5), para correspondência zero-shot de pacientes a ensaios clínicos. Ele automatiza a triagem de elegibilidade, dividindo a tarefa em três módulos: 1) **TrialGPT-Retrieval** para filtragem em larga escala e geração de palavras-chave; 2) **TrialGPT-Matching** para predição de elegibilidade em nível de critério com explicações fiéis; e 3) **TrialGPT-Ranking** para pontuação e classificação final dos ensaios.

## Statistics

Avaliado em três coortes (SIGIR 2016, TREC 2021/2022) com 183 pacientes sintéticos e mais de 75.000 anotações. **TrialGPT-Retrieval** alcança mais de 90% de recall usando menos de 6% da coleção inicial. **TrialGPT-Matching** atinge 87.3% de precisão em 1015 pares paciente-critério, comparável a especialistas. O uso do TrialGPT demonstrou uma redução de 42.6% no tempo de triagem de pacientes.

## Features

Correspondência zero-shot (sem fine-tuning específico para cada ensaio). Explicabilidade (gera explicações fiéis para as decisões de elegibilidade). Escalabilidade para grandes coleções de ensaios clínicos (até 23.000 ensaios ativos). LLM-agnóstico (pode ser adaptado a diferentes LLMs).

## Use Cases

Triagem e recrutamento automatizado de pacientes para ensaios clínicos. Redução da carga de trabalho manual de triagem. Priorização de ensaios clínicos mais relevantes para um paciente específico.

## Integration

O framework utiliza prompts de Chain-of-Thought (CoT) para as tarefas de Matching e Ranking. Os prompts são estruturados para incluir: 1) Descrição da tarefa; 2) Informações de background clínico; 3) Critérios de inclusão e exclusão. O objetivo é gerar uma explicação de relevância (R), uma lista de IDs de sentenças relevantes (S) e a predição de elegibilidade (E) para cada critério. Exemplo de estrutura de prompt (adaptado do artigo): 'Você é um especialista em elegibilidade de ensaios clínicos. Dada a nota do paciente e o critério de elegibilidade [CRITÉRIO], determine se o paciente é 'incluído', 'não incluído', 'informação insuficiente' ou 'não aplicável'. Pense passo a passo e forneça a explicação, as sentenças de evidência e a decisão final no formato JSON.'

## URL

https://www.nature.com/articles/s41467-024-53081-z