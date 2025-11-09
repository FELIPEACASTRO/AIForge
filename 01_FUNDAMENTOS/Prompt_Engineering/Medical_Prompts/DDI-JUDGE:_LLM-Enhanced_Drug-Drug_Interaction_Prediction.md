# DDI-JUDGE: LLM-Enhanced Drug-Drug Interaction Prediction

## Description

O **DDI-JUDGE** é um método inovador que utiliza Large Language Models (LLMs) para a predição de Interações Medicamentosas (DDI), integrando a **Aprendizagem em Contexto (ICL)** e um módulo de **"Julgamento" (Judging)**. O método propõe um novo paradigma de prompt ICL que seleciona amostras de alta similaridade molecular para otimizar o aprendizado do modelo. Além disso, emprega um LLM discriminador (como o GPT-4) para avaliar a qualidade das explicações geradas por múltiplos LLMs e agrega os resultados por meio de votação ponderada, aumentando a robustez e a precisão da predição. Este modelo representa um avanço significativo na aplicação de LLMs para tarefas de segurança farmacológica.

## Statistics

**Métricas de Desempenho (Zero-shot / Few-shot):**
- **AUC (Area Under the Curve):** 0.642 / 0.788
- **AUPR (Area Under the Precision-Recall Curve):** 0.629 / 0.801
- O método demonstrou desempenho superior em comparação com outros métodos baseados em LLM em ambos os cenários de aprendizado (zero-shot e few-shot).
- **Citação:** Qi et al. (2025). Improving drug-drug interaction prediction via in-context learning and judging with large language models. *Frontiers in Pharmacology*.

## Features

- **ICL Otimizado:** Seleção de amostras ICL (positivas e negativas) baseada em similaridade molecular (Tanimoto, Cosine, Dice) para guiar o LLM.
- **Módulo Discriminador (GPT-4):** Avalia a precisão científica, clareza, suporte de evidências e relevância das explicações geradas por outros LLMs.
- **Agregação de Resultados:** Combinação de previsões de múltiplos LLMs usando um esquema de votação ponderada pela confiança.
- **Estrutura Modular:** Facilmente aplicável em fluxos de trabalho clínicos e outras aplicações biomédicas.

## Use Cases

- **Predição de DDI em Polifarmácia:** Identificação de interações medicamentosas em pacientes que utilizam múltiplos fármacos.
- **Suporte à Decisão Clínica:** Auxílio a farmacêuticos e médicos na avaliação rápida e fundamentada de riscos de DDI.
- **Triagem de Segurança Farmacológica:** Aplicação em estágios iniciais de descoberta de medicamentos para prever potenciais interações adversas.

## Integration

**Guia de Integração (Prompt Zero-Shot):**

O prompt é estruturado para simular uma consulta a um farmacologista experiente, exigindo uma análise baseada em fatores farmacológicos e evidências.

**Instrução de Sistema/Papel:**
"You are an experienced pharmacologist with extensive knowledge of drug interactions. Your task is to determine whether there is an interaction between the two drugs based on their pharmacological profiles. Specifically, you should consider the following factors:
- Pharmacodynamics: How the drugs affect the body, including their effects on receptors and physiological systems.
- Metabolic Pathways: How the drugs are metabolized, including enzyme interactions and potential effects on drug metabolism.
- Receptor Interactions: Whether the drugs interact with the same or similar receptors.
Task: Given the names and SMILES structures of two drugs, predict if there is an interaction between them. You should answer "yes" if there is an adverse interaction and "no" if there is no adverse interaction."

**Requisitos de Saída:**
"Requirements:
- Prediction and Explanation: Provide a binary prediction ("yes" for an interaction or "no" for no interaction) and a concise explanation grounded in pharmacological evidence.
- Evidence-Based: Use reliable sources such as clinical trials, FDA labeling, drug interaction databases, and peer-reviewed literature to support your explanation.
- Structured Explanation: Clearly outline the reasoning for your prediction, addressing the pharmacological factors explicitly."

**Exemplo de Pergunta (Template):**
```
Drug A Name: [Nome do Medicamento A]
Drug A Smiles: [Estrutura SMILES do Medicamento A]
Drug B Name: [Nome do Medicamento B]
Drug B Smiles: [Estrutura SMILES do Medicamento B]
Interaction Prediction: [yes/no]
Explanation: [Fornecer uma explicação detalhada referenciando farmacodinâmica, vias metabólicas, interações de receptores e dados clínicos.]
```

## URL

https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1589788/full