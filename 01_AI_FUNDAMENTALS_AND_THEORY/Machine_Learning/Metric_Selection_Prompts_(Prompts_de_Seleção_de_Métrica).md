# Metric Selection Prompts (Prompts de Seleção de Métrica)

## Description
"Metric Selection Prompts" (Prompts de Seleção de Métrica) é uma técnica avançada de Engenharia de Prompt que se concentra em **integrar explicitamente os critérios de avaliação e as métricas de sucesso** dentro do próprio prompt. Em vez de apenas instruir o Modelo de Linguagem Grande (LLM) sobre a tarefa, o prompt também o instrui sobre **como** a saída deve ser medida e avaliada. Essa técnica é fundamental no campo de **Avaliação de LLMs (LLM Evaluation)** e no desenvolvimento de **Modelos como Juízes (Model-as-a-Judge)**, onde um LLM é usado para avaliar a qualidade da saída de outro LLM com base em métricas específicas como **Relevância**, **Coerência**, **Fluência**, **Groundedness** (Aderência à Fonte) e **Similaridade** [1] [2]. O objetivo é garantir que a saída do modelo não apenas complete a tarefa, mas também atenda a um padrão de qualidade mensurável e predefinido, alinhando o desempenho do modelo com os objetivos de negócios ou técnicos [3].

## Examples
```
**Exemplo 1: Avaliação de Groundedness (Aderência à Fonte)**
```
**Instrução:** Você é um avaliador de LLM. Sua tarefa é determinar a "Groundedness" da Resposta Gerada em relação ao Contexto Fornecido.
**Contexto:** [Trecho de um documento de origem]
**Pergunta:** [Pergunta original do usuário]
**Resposta Gerada:** [Saída do LLM a ser avaliada]
**Métrica de Avaliação (Groundedness):** A resposta é verificável e suportada *apenas* pelo Contexto Fornecido?
**Escala:** 1 (Nenhuma parte é suportada) a 5 (Toda a resposta é diretamente suportada).
**Saída (JSON):** {"Groundedness_Score": [1-5], "Justificativa": "..."}
```

**Exemplo 2: Avaliação de Relevância**
```
**Instrução:** Avalie a "Relevância" da Resposta Gerada para a Pergunta Original.
**Pergunta:** Qual é o impacto da IA generativa no setor financeiro?
**Resposta Gerada:** A IA generativa está transformando o setor financeiro ao automatizar a detecção de fraudes e personalizar o atendimento ao cliente.
**Métrica de Avaliação (Relevância):** A resposta aborda diretamente o tópico da pergunta e captura os pontos-chave?
**Escala:** 1 (Irrelevante) a 5 (Perfeitamente Relevante).
**Saída:** Pontuação de Relevância: [1-5].
```

**Exemplo 3: Otimização para Coerência e Fluência**
```
**Instrução:** Reescreva o texto a seguir para maximizar a "Coerência" e a "Fluência".
**Texto Original:** "O setor de saúde, a IA, está mudando. Os diagnósticos agora são mais rápidos, e os pacientes, o tratamento, é personalizado."
**Métrica de Otimização:** A saída deve ter fluxo suave, gramática impecável e transições lógicas, como se fosse escrita por um especialista humano.
**Saída:** [Texto reescrito otimizado para Coerência e Fluência]
```

**Exemplo 4: Seleção de Métrica para Tarefa de Resumo**
```
**Instrução:** Resuma o artigo abaixo em 3 frases. A métrica de sucesso para este resumo é a **Completeness (Abrangência)**.
**Artigo:** [Corpo do artigo]
**Critério de Sucesso (Completeness):** O resumo deve cobrir os 3 pontos principais do artigo.
**Saída:** [Resumo de 3 frases]
```

**Exemplo 5: Prompt para Métrica de Aderência ao Formato**
```
**Instrução:** Gere uma lista de 5 ideias de títulos para um blog post sobre "Prompt Engineering".
**Métrica de Avaliação (Aderência ao Formato):** Cada título deve ter no máximo 60 caracteres e ser formatado como uma lista numerada.
**Saída:**
1. [Título 1]
2. [Título 2]
3. [Título 3]
4. [Título 4]
5. [Título 5]
```

**Exemplo 6: Prompt para Métrica de Tom (Subjetiva)**
```
**Instrução:** Responda à reclamação do cliente com uma mensagem de desculpas.
**Métrica de Avaliação (Tom):** O tom da resposta deve ser **Empático** (pontuação 5/5) e **Profissional** (pontuação 5/5). Evite linguagem excessivamente formal ou robótica.
**Saída:** [Resposta ao cliente]
```

**Exemplo 7: Prompt para Métrica de Similaridade (Parafraseamento)**
```
**Instrução:** Parafraseie a frase a seguir.
**Frase Original:** "A inteligência artificial está revolucionando a forma como interagimos com a tecnologia."
**Métrica de Avaliação (Similaridade Semântica):** A saída deve manter o significado central da frase original, mas usar vocabulário e estrutura de frase completamente diferentes.
**Saída:** [Frase parafraseada]
```
```

## Best Practices
**1. Defina a Métrica Antes do Prompt:** Antes de escrever o prompt, determine a métrica de sucesso (ex: "Acurácia", "Relevância", "Coerência"). O prompt deve ser desenhado para otimizar essa métrica.
**2. Incorpore o Critério de Avaliação:** Inclua no prompt uma seção clara que defina como a saída será avaliada. Por exemplo, "A resposta deve ser avaliada em uma escala de 1 a 5 para 'Groundedness' (Aderência à Fonte)".
**3. Use Modelos Avaliadores (Model-as-a-Judge):** Em vez de apenas gerar a resposta, use o prompt para instruir um LLM a atuar como um avaliador, comparando a saída de outro modelo com o critério de sucesso.
**4. Forneça Contexto e Base de Fato (Ground Truth):** Para métricas como "Groundedness" e "Relevância", o prompt deve incluir o contexto de referência (documentos, trechos) para que o modelo possa verificar a fidelidade da informação.
**5. Padronize a Saída para Avaliação:** Peça ao modelo para formatar a saída de forma estruturada (JSON, XML) que inclua a resposta gerada e a pontuação da métrica, facilitando a análise automatizada.

## Use Cases
**1. Avaliação Automatizada de Prompts (Prompt Evaluation):** Usar um LLM como juiz para avaliar a qualidade da saída de outro LLM em larga escala, substituindo ou complementando a avaliação humana (Human-in-the-Loop) [2].
**2. Desenvolvimento de RAG (Retrieval-Augmented Generation):** Otimizar prompts para métricas como **Groundedness** (Aderência à Fonte) e **Relevância** para garantir que as respostas sejam factualmente corretas e baseadas nos documentos recuperados [1].
**3. Teste A/B de Versões de Prompt:** Comparar diferentes versões de prompts (V1 vs. V2) usando métricas objetivas (ex: Acurácia, Latência) para determinar qual versão oferece o melhor desempenho para um objetivo específico [3].
**4. Monitoramento de Modelos em Produção:** Integrar métricas de avaliação no fluxo de trabalho de monitoramento para detectar desvios de qualidade (drift) na saída do LLM ao longo do tempo (ex: queda na Coerência ou Relevância) [4].
**5. Geração de Dados Sintéticos de Alta Qualidade:** Usar prompts que exigem alta pontuação em métricas como **Fluência** e **Coerência** para gerar dados de treinamento sintéticos que imitam a qualidade da linguagem humana.

## Pitfalls
**1. Métrica Mal Definida:** Usar termos vagos como "boa qualidade" ou "melhor resposta" em vez de métricas quantificáveis (ex: "Acurácia Factual > 90%").
**2. Conflito de Métricas:** Pedir ao modelo para otimizar métricas que se anulam, como exigir "Máxima Criatividade" e "Máxima Aderência a Regras Estritas" no mesmo prompt.
**3. Falha na Padronização da Saída:** Não instruir o modelo a formatar a pontuação da métrica de forma estruturada (JSON, XML), dificultando a coleta e análise automatizada dos resultados de avaliação.
**4. Ausência de Contexto (Ground Truth):** Tentar avaliar métricas baseadas em fatos (como Groundedness) sem fornecer o contexto de origem para o modelo avaliador.
**5. Confiança Excessiva em Métricas Subjetivas:** Depender apenas de métricas subjetivas (como "Coerência" ou "Fluência") sem validação humana ou sem um prompt de avaliação muito detalhado e robusto.
**6. Ignorar o Custo Computacional:** O uso de LLMs como juízes (Model-as-a-Judge) para avaliar métricas aumenta significativamente o custo e a latência da aplicação, um fator a ser considerado no design do prompt.

## URL
[https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2)
