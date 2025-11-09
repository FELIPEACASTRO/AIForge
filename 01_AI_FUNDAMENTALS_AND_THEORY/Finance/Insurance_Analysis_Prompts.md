# Insurance Analysis Prompts

## Description
A engenharia de prompts para análise de seguros refere-se à criação e otimização de instruções (prompts) para Large Language Models (LLMs) com o objetivo de automatizar e aprimorar tarefas específicas do setor de seguros. Isso inclui a análise de documentos de apólices, processamento de sinistros, avaliação de risco (subscrição), detecção de fraude e interação com o cliente. A chave é fornecer ao LLM contexto e estrutura suficientes para que ele atue como um especialista em seguros, garantindo precisão e conformidade regulatória. A precisão e a conformidade regulatória são aspectos críticos que exigem o uso de técnicas avançadas de prompt, como o Chain-of-Thought (CoT), para garantir a rastreabilidade e a justificativa das decisões.

## Examples
```
### Exemplo 1: Sumarização de Apólice (Policy Summarization)
**Papel:** Analista de Apólices Sênior
**Instrução:** "Você é um Analista de Apólices Sênior. Sua tarefa é resumir a apólice de seguro fornecida abaixo. O resumo deve ter no máximo 200 palavras e deve destacar obrigatoriamente: 1) O limite máximo de cobertura para 'Danos a Terceiros', 2) O valor da franquia (dedutível) para 'Colisão', e 3) As duas principais exclusões de cobertura. Use a seção exata da apólice para justificar cada ponto."
**Entrada:** `[Texto completo da apólice]`

### Exemplo 2: Avaliação de Sinistro (Claim Evaluation)
**Papel:** Avaliador de Sinistros
**Instrução:** "Atue como um Avaliador de Sinistros. Analise os 'Detalhes do Sinistro' e o 'Trecho da Apólice' fornecidos. Determine se o sinistro é coberto. **Passos de Raciocínio (Chain-of-Thought):** 1. Identifique a cláusula de cobertura relevante. 2. Verifique se alguma exclusão se aplica. 3. Conclua a elegibilidade. Sua resposta final deve ser 'COBERTO' ou 'NÃO COBERTO', seguida pela sua justificativa passo a passo."
**Entrada:** `Detalhes do Sinistro: [Descrição do evento]`, `Trecho da Apólice: [Cláusulas relevantes]`

### Exemplo 3: Comparação de Propostas (Proposal Comparison)
**Papel:** Consultor de Seguros
**Instrução:** "Você é um Consultor de Seguros imparcial. Compare as duas propostas de seguro de automóvel abaixo. Crie uma tabela comparativa que inclua: Prêmio Anual, Cobertura contra Roubo, Franquia e Classificação de Risco (Baixo, Médio, Alto) com base nas exclusões. Recomende a melhor opção para um cliente que prioriza o menor custo total em caso de sinistro."
**Entrada:** `Proposta A: [Detalhes]`, `Proposta B: [Detalhes]`

### Exemplo 4: Análise de Risco de Subscrição (Underwriting Risk Analysis)
**Papel:** Subscritor de Risco
**Instrução:** "Analise o perfil do proponente e o histórico de sinistros. Com base nos dados, atribua uma pontuação de risco de 1 (Baixo) a 10 (Alto) e justifique a pontuação. Se a pontuação for superior a 7, sugira uma cláusula de exclusão ou um aumento de prêmio de 15%."
**Entrada:** `Perfil do Proponente: [Idade, Localização, Tipo de Propriedade]`, `Histórico de Sinistros: [Data, Tipo, Valor]`

### Exemplo 5: Geração de Comunicação de Não Cobertura (Denial Letter Generation)
**Papel:** Agente de Comunicação Legal
**Instrução:** "Gere uma carta formal de recusa de sinistro para o cliente [Nome do Cliente]. O motivo da recusa é a 'Cláusula de Exclusão de Desastres Naturais' (Seção 4.B da apólice). A carta deve ser empática, citar a seção exata da apólice e informar o cliente sobre o processo de apelação."
**Entrada:** `Nome do Cliente: João Silva`, `Número da Apólice: 12345`, `Data do Sinistro: 01/01/2025`
```

## Best Practices
1. **Atribuição de Papel (Role Assignment):** Comece o prompt definindo o papel do LLM (ex: "Você é um subscritor sênior de seguros de propriedade e acidentes").
2. **Estrutura e Variáveis:** Utilize templates de prompt estruturados com variáveis claras (ex: `{policy_text}`, `{claim_details}`) para garantir que o modelo receba todas as informações necessárias.
3. **Estratégias de Raciocínio:** Implemente estratégias de prompting avançadas como **Chain-of-Thought (CoT)** para tarefas complexas (ex: avaliação de sinistros), solicitando que o modelo justifique seu raciocínio passo a passo.
4. **Foco em Conformidade e Precisão:** Inclua instruções explícitas para que o modelo cite a seção exata da apólice ou regulamento que suporta sua conclusão.
5. **Avaliação Rigorosa:** Utilize frameworks de avaliação com métricas específicas do domínio (ex: precisão na extração de cláusulas, conformidade regulatória) e avaliação humana.

## Use Cases
1. **Subscrição e Avaliação de Risco:** Analisar dados de candidatos a seguro (histórico de sinistros, informações demográficas) para determinar a elegibilidade e o prêmio.
2. **Processamento de Sinistros:** Avaliar a validade de um sinistro, calcular o valor do pagamento e gerar a carta de decisão, comparando os detalhes do sinistro com os termos da apólice.
3. **Análise de Documentos de Apólice:** Sumarizar apólices complexas, extrair cláusulas específicas (exclusões, franquias) e comparar diferentes propostas de seguro.
4. **Detecção de Fraude:** Analisar padrões em relatórios de sinistros e documentos de suporte para identificar anomalias ou indicadores de fraude.
5. **Atendimento ao Cliente (Chatbots Especializados):** Responder a perguntas complexas de clientes sobre coberturas e processos de sinistro com base em documentos internos.

## Pitfalls
1. **Alucinações e Imprecisão:** O LLM pode gerar informações factualmente incorretas ou citar cláusulas inexistentes, o que é crítico em um setor regulamentado.
2. **Viés e Injustiça:** Se os dados de treinamento contiverem viés, o modelo pode levar a decisões de subscrição ou sinistro injustas ou discriminatórias.
3. **Falta de Contexto Específico:** Prompts muito genéricos falham em capturar a complexidade e a terminologia técnica do domínio de seguros.
4. **Dependência Excessiva de Zero-Shot:** Para tarefas de alto risco, depender apenas de prompts simples (zero-shot) sem CoT ou validação externa é perigoso.
5. **Insegurança de Dados (Data Leakage):** O uso de dados sensíveis de clientes em prompts sem a devida anonimização ou em modelos não seguros.

## URL
[https://github.com/ozturkoktay/insurance-llm-framework](https://github.com/ozturkoktay/insurance-llm-framework)
