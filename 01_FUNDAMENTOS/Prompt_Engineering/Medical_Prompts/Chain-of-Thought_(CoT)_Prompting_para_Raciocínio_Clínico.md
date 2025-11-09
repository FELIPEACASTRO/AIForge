# Chain-of-Thought (CoT) Prompting para Raciocínio Clínico

## Description

O **Chain-of-Thought (CoT) Prompting** é uma técnica de engenharia de prompt que instrui o Large Language Model (LLM) a gerar uma série de passos intermediários de raciocínio antes de fornecer a resposta final. Este método visa aumentar a **interpretabilidade** e o desempenho em tarefas de raciocínio complexo. No contexto do raciocínio clínico, sua eficácia é mista. Enquanto o CoT Tradicional demonstrou resultados estáveis e melhorias em tarefas estruturadas de Resposta a Perguntas Médicas (Medical Question Answering), sua aplicação em textos clínicos não estruturados (como Registros Eletrônicos de Saúde - EHRs) demonstrou ser problemática. Estudos recentes (2025) indicam que o CoT pode sistematicamente prejudicar a acurácia na compreensão de texto clínico real, com 86.3% dos modelos avaliados sofrendo degradação de desempenho. As falhas são atribuídas a cadeias de raciocínio mais longas e a um aterramento (grounding) mais fraco em conceitos clínicos, levantando preocupações sobre a fidelidade e o risco de *over-trust* nas explicações geradas.

## Statistics

- **Acurácia Variável:** Em tarefas de Resposta a Perguntas Médicas (QA), o CoT Tradicional alcançou até **88.4%** de acurácia no dataset EHRNoteQA (usando o modelo o1-mini). Em tarefas mais complexas (MedMCQA), o CoT Interativo caiu para **61.7%**.
- **Degradação em Texto Clínico Real:** Um estudo de 2025 avaliando 95 LLMs em 87 tarefas clínicas multilíngues demonstrou que **86.3%** dos modelos sofreram degradação de desempenho consistente ao usar CoT em tarefas de compreensão de texto clínico não estruturado (EHRs).
- **Modelos Avaliados:** o1-mini (melhor desempenho geral com CoT), GPT-4o-mini, Gemini-1.5-Flash, GPT-3.5-turbo, Gemini-1.0-pro.
- **Citação:** Jeon et al. (2025), Wu et al. (2025).

## Features

- **Aumento da Interpretabilidade:** Torna o processo de raciocínio do LLM transparente.
- **Melhoria de Desempenho:** Potencial para elevar a acurácia em tarefas de raciocínio complexo e estruturadas (e.g., MedQA).
- **Variações:** Inclui CoT Tradicional, Zero-Shot CoT, Few-Shot CoT, Interactive CoT, ReAct CoT e Self-Consistency CoT.
- **Limitação Crítica:** Desempenho inconsistente ou degradado em textos clínicos não estruturados e ruidosos (EHRs).

## Use Cases

- **Resposta a Perguntas Médicas (MQA):** Auxílio na resolução de questões de múltipla escolha e cenários clínicos estruturados (e.g., MedQA, MedMCQA).
- **Suporte à Decisão Clínica:** Melhoria da interpretabilidade em sistemas de suporte à decisão, permitindo que o médico revise o processo de raciocínio do LLM.
- **Educação Médica:** Treinamento de estudantes e residentes, fornecendo um passo a passo lógico para o diagnóstico e manejo de casos.

## Integration

**Exemplo de Prompt para Raciocínio Clínico:**

```
Pense passo a passo. Um paciente de 65 anos apresenta-se com dor torácica súbita, dispneia e histórico de tabagismo. O ECG mostra elevação do segmento ST em V1-V4. Qual é o diagnóstico mais provável? Justifique seu raciocínio clínico detalhadamente antes de dar a resposta final.
```

**Instrução de Integração:** O modelo deve ser instruído a gerar uma "cadeia de pensamento" (ex: "1. Analisar sintomas... 2. Analisar histórico... 3. Analisar ECG... 4. Concluir o diagnóstico.") antes de fornecer a resposta final. A técnica **CoT Tradicional** mostrou-se a mais consistente em avaliações de QA médica. É crucial validar a saída do CoT em ambientes clínicos reais devido ao risco de alucinação e degradação de acurácia.

## URL

https://www.sciencedirect.com/science/article/pii/S0010482525009655