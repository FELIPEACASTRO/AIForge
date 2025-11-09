# Radiology Interpretation Prompts (Prompts de Interpretação Radiológica)

## Description
**Prompts de Interpretação Radiológica** são instruções de engenharia de prompt (Prompt Engineering) especificamente desenhadas para orientar Modelos de Linguagem Grande (LLMs), como GPT-4 ou Gemini, na análise e geração de conteúdo relacionado a exames de imagem médica (radiografias, tomografias, ressonâncias magnéticas, etc.). O objetivo principal é transformar dados brutos de imagem ou achados descritivos em **laudos estruturados, resumos para pacientes, auxílio diagnóstico ou ferramentas de ensino** [1] [2].

A técnica se baseia em fornecer ao LLM um **papel** (ex: radiologista), o **contexto clínico** e os **achados da imagem**, solicitando uma saída em um **formato estruturado e profissional** [2]. Estudos recentes (2024-2025) demonstram que prompts bem elaborados, que incluem raciocínio passo a passo e limites de confiança, podem **melhorar significativamente a precisão diagnóstica** dos LLMs em casos complexos, como em neurorradiologia [3].

Os prompts eficazes em radiologia são caracterizados por quatro elementos chave: **Instruções/Perguntas**, **Contexto**, **Entrada de Dados** e **Formato de Saída** [2]. A aplicação correta desta técnica é crucial para integrar LLMs de forma segura e eficiente no fluxo de trabalho clínico.

## Examples
```
**1. Geração de Laudo Estruturado (Raio-X de Tórax)**

```
**Papel:** Você é um radiologista torácico experiente.
**Instrução:** Analise os achados descritivos do Raio-X de Tórax e gere um laudo estruturado e conciso.
**Achados:** "Opacidade alveolar no lobo inferior direito, associada a broncograma aéreo. Silhueta cardíaca e mediastino sem alterações. Seios costofrênicos livres."
**Formato de Saída:**
**Achados:** [Achados detalhados]
**Impressão:** [Diagnóstico mais provável e diagnóstico diferencial]
**Recomendação:** [Sugestão de acompanhamento ou exame complementar]
```

**2. Resumo de Laudo para Paciente (Linguagem Leiga)**

```
**Papel:** Você é um assistente de comunicação médica.
**Instrução:** Resuma o laudo radiológico abaixo em linguagem simples e acessível para um paciente, focando apenas nos achados mais importantes e nas implicações. Mantenha um tom tranquilizador e informativo.
**Laudo:** "RM de crânio: Lesão expansiva intra-axial, com realce anelar e necrose central, localizada no lobo temporal esquerdo. Sugere glioblastoma multiforme (GBM). Necessário correlação com biópsia e acompanhamento oncológico."
**Formato de Saída:** Título (O que o exame mostrou), Resumo (Explicação simples), Próximos Passos (O que fazer a seguir).
```

**3. Auxílio Diagnóstico Diferencial (Tomografia de Abdômen)**

```
**Papel:** Você é um consultor de diagnóstico por imagem.
**Instrução:** Com base nos achados da TC de Abdômen, forneça uma lista de 3 diagnósticos diferenciais mais prováveis, juntamente com o nível de confiança (Alto, Médio, Baixo) para cada um e o raciocínio clínico que os suporta.
**Achados:** "Massa sólida, heterogênea, com calcificações grosseiras e realce tardio no polo superior do rim direito. Não há linfonodomegalia retroperitoneal."
**Formato de Saída:** Tabela com colunas: Diagnóstico, Confiança, Raciocínio.
```

**4. Geração de Texto de Discussão para Artigo Científico**

```
**Papel:** Você é um pesquisador em radiologia.
**Instrução:** Escreva a seção de Discussão de um artigo científico sobre o uso de IA na detecção precoce de nódulos pulmonares. Inclua referências a estudos recentes (2023-2025) e discuta as limitações e o futuro da tecnologia.
**Tópicos Chave:** Desempenho da IA (AUC > 0.95), Redução de falsos positivos, Desafio da generalização, Integração PACS.
**Formato de Saída:** Texto acadêmico, parágrafos bem estruturados, com citações no formato [N].
```

**5. Otimização de Protocolo de Exame (Ressonância Magnética)**

```
**Papel:** Você é um técnico de RM sênior.
**Instrução:** Sugira otimizações para o protocolo de RM de Joelho para melhor visualização da cartilagem articular, considerando um equipamento de 3T.
**Protocolo Atual:** T1, T2, PD-FS.
**Otimização Desejada:** Melhorar a resolução espacial e o contraste da cartilagem.
**Formato de Saída:** Lista de sequências sugeridas (ex: 3D DESS, T2 Mapping) e justificativa técnica para cada uma.
```
```

## Best Practices
**1. Clareza e Precisão (Clarity and Precision):** O prompt deve ser **claro, conciso e inequívoco**, definindo a tarefa, o público-alvo (ex: radiologista, paciente) e o formato de saída desejado (ex: laudo estruturado, resumo para leigo) [1] [2].
**2. Adoção de Papel (Role Adoption):** Comece o prompt instruindo o LLM a assumir um papel específico, como "Você é um radiologista especializado em neurorradiologia" ou "Você é um assistente de transcrição médica" [3].
**3. Contexto Relevante (Relevant Context):** Forneça o máximo de contexto clínico e de imagem possível. Isso inclui o tipo de exame (TC, RM, RX), a área anatômica, a história clínica do paciente e os achados brutos da imagem [1].
**4. Raciocínio Passo a Passo (Step-by-Step Reasoning):** Para tarefas complexas, como diagnóstico diferencial, instrua o LLM a usar o método "Cadeia de Pensamento" (Chain-of-Thought - CoT) ou "Raciocínio Clínico Estruturado" (Structured Clinical Reasoning) [3]. Peça para listar os achados, correlacionar com patologias e, por fim, fornecer a impressão diagnóstica.
**5. Limitação de Saída (Output Constraint):** Especifique o formato de saída (ex: JSON, lista com marcadores, texto corrido) e a estrutura (ex: "Achados", "Impressão", "Recomendação"). Isso garante a consistência e facilita a integração com sistemas de relatórios [2].
**6. Limite de Confiança (Confidence Threshold):** Em prompts de auxílio diagnóstico, peça ao LLM para fornecer um nível de confiança para cada diagnóstico sugerido. Isso ajuda a filtrar sugestões de baixa probabilidade e aumenta a segurança [3].
**7. Iteração e Refinamento (Iteration and Refinement):** O design do prompt deve ser um processo iterativo. Teste e ajuste o prompt com base na precisão e relevância das respostas do LLM [1].

## Use Cases
nan

## Pitfalls
**1. Alucinação e Invenção de Achados:** O LLM pode "alucinar" (inventar) achados clínicos ou diagnósticos que não estão presentes nos dados de entrada. **Mitigação:** Sempre instrua o modelo a basear a interpretação **apenas** nos dados fornecidos e a indicar quando a informação é insuficiente [1].
**2. Falta de Contexto Clínico:** Prompts muito curtos ou sem o contexto clínico adequado (idade, sexo, sintomas, histórico) levam a diagnósticos genéricos ou incorretos. **Mitigação:** Inclua sempre um bloco de "História Clínica" no prompt [2].
**3. Viés e Generalização:** LLMs treinados em dados de uma população específica podem apresentar viés ao interpretar exames de outras populações. **Mitigação:** Não há solução direta via prompt, mas o radiologista deve estar ciente e usar o LLM apenas como auxílio, não como decisão final.
**4. Inconsistência de Formato:** A falha em especificar um formato de saída rigoroso (ex: JSON ou tabela) pode resultar em respostas inconsistentes e difíceis de integrar em sistemas de informação hospitalar (HIS/RIS). **Mitigação:** Use o elemento "Formato de Saída" de forma explícita e com exemplos [2].
**5. Confiança Excessiva:** A confiança na saída do LLM, especialmente em casos complexos, pode levar a erros de diagnóstico. **Mitigação:** Use o LLM como uma "segunda opinião" ou assistente de rascunho, e não como o autor final do laudo. A revisão humana é obrigatória [3].

## URL
[https://www.jacr.org/article/S1546-1440(25)00156-5/fulltext](https://www.jacr.org/article/S1546-1440(25)00156-5/fulltext)
