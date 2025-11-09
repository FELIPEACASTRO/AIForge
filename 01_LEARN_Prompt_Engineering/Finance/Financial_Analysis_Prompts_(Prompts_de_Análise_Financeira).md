# Financial Analysis Prompts (Prompts de Análise Financeira)

## Description
A técnica de **Prompts de Análise Financeira** refere-se à arte e ciência de criar instruções e perguntas otimizadas para modelos de linguagem grande (LLMs) com o objetivo de realizar tarefas complexas de finanças, contabilidade, investimento e risco. Envolve fornecer contexto financeiro específico, dados estruturados (como balanços, demonstrações de resultados ou dados de mercado) e definir o formato de saída desejado (como relatórios, resumos ou previsões). A eficácia reside na capacidade de transformar dados brutos em *insights* acionáveis, automatizando a análise de tendências, a avaliação de riscos, o planejamento de cenários e a conformidade regulatória. É uma aplicação crítica da Engenharia de Prompt, pois a precisão e a conformidade são essenciais no setor financeiro.

## Examples
```
1. **Análise de Demonstrações Financeiras:** "Com base no Balanço Patrimonial e na Demonstração de Resultados anexados da Empresa X (2023-2024), atue como um Analista de Crédito Sênior. Calcule e interprete os seguintes índices: Liquidez Corrente, Endividamento Geral e Margem Líquida. Apresente os resultados em uma tabela Markdown e forneça um resumo de 2 parágrafos sobre a saúde financeira da empresa."

2. **Planejamento de Cenários (Stress Test):** "Considerando o portfólio de investimentos em anexo (Ações: 60%, Títulos: 30%, Ouro: 10%), simule o impacto financeiro de um cenário de 'crise inflacionária' (inflação de 15%, queda de 20% no mercado de ações). Qual seria a perda percentual do portfólio? Sugira 3 ações defensivas para mitigar esse risco."

3. **Otimização de Orçamento:** "Analise o orçamento do Departamento de Marketing do último trimestre. Identifique as 3 maiores áreas de despesa e sugira 2 áreas onde um corte de 10% seria mais viável, justificando o impacto operacional de cada corte. O resultado deve ser um relatório conciso."

4. **Análise de Conformidade (Compliance):** "Revise o extrato de transações de alto valor do mês passado. Identifique qualquer transação que possa levantar uma 'bandeira vermelha' sob a regulamentação de Prevenção à Lavagem de Dinheiro (PLD), especificando o motivo da suspeita e o próximo passo regulatório recomendado."

5. **Previsão de Fluxo de Caixa:** "Usando os dados históricos de fluxo de caixa dos últimos 6 meses (média de entrada: R$ 500k/mês, média de saída: R$ 450k/mês, com pico de saída em Dezembro de R$ 600k), projete o saldo de caixa para os próximos 3 meses. Inclua uma análise de sensibilidade para uma queda de 10% nas receitas."

6. **Avaliação de Ativos (M&A):** "Atue como um Consultor de M&A. Avalie a viabilidade financeira da aquisição da 'Startup Y' (Receita Anual: R$ 5M, Lucro Líquido: R$ 500k, Dívida Total: R$ 1M). Use o método de Múltiplos de Receita (5x) e sugira um preço de compra justo, listando 3 riscos financeiros chave da transação."

7. **Interpretação de Indicadores de Mercado:** "Explique o que o índice P/L (Preço/Lucro) de 25x da Empresa Z significa para um investidor de longo prazo, comparando-o com a média do setor (15x). A explicação deve ser clara e didática, adequada para um investidor iniciante."
```

## Best Practices
**1. Fornecer Dados Estruturados:** Sempre inclua dados financeiros reais (tabelas, CSVs, ou listas) diretamente no prompt ou referencie um documento/fonte de dados. A precisão da análise depende da qualidade e do formato dos dados de entrada. **2. Definir o Papel (Persona):** Comece o prompt definindo o LLM como um "Analista Financeiro Sênior", "Contador Certificado" ou "Especialista em Risco", para direcionar o tom e a profundidade da resposta. **3. Especificar o Formato de Saída:** Peça a saída em um formato claro e utilizável, como "Tabela Markdown", "Resumo Executivo de 3 Parágrafos" ou "JSON com os KPIs". **4. Incluir Restrições e Premissas:** Para análises de cenário ou risco, defina claramente as premissas (ex: "Considere um aumento de 15% na taxa Selic") para garantir que a análise seja relevante. **5. Solicitar Fontes e Justificativas:** Peça ao LLM para citar as fontes de dados ou justificar a metodologia de cálculo para garantir a rastreabilidade e a confiança nos resultados.

## Use Cases
**1. Planejamento Estratégico e Orçamentário:** Criação de orçamentos anuais, projeções de receita e despesas, e análise de sensibilidade para diferentes cenários econômicos. **2. Gestão de Risco e Conformidade (Compliance):** Avaliação de risco de crédito, detecção de fraudes em transações, monitoramento de atividades financeiras para aderência a regulamentações (ex: IFRS, BACEN, CVM). **3. Investimento e Avaliação (Valuation):** Análise de portfólios de ações, avaliação de ativos para fusões e aquisições (M&A), e *due diligence* financeira. **4. Contabilidade e Relatórios:** Geração automatizada de resumos executivos de demonstrações financeiras, categorização de despesas e criação de relatórios de desempenho para stakeholders. **5. Otimização de Processos:** Análise de folha de pagamento para otimização de custos e automação de tarefas repetitivas de análise de dados.

## Pitfalls
**1. Alucinações de Dados:** O LLM pode inventar dados financeiros, índices ou regulamentações. **Contramedida:** Sempre peça a citação da fonte ou a fórmula de cálculo. **2. Falta de Contexto:** Prompts muito genéricos sem especificar o papel, o público-alvo ou o período de tempo levam a respostas superficiais. **Contramedida:** Use a técnica de *Persona* e *Restrição*. **3. Viés e Simplificação Excessiva:** LLMs podem simplificar demais análises complexas de risco ou ignorar nuances regulatórias. **Contramedida:** Peça uma análise de "múltiplos cenários" ou a inclusão de "riscos não-quantificáveis". **4. Vazamento de Informações Confidenciais:** Inserir dados financeiros sensíveis em plataformas públicas de LLM. **Contramedida:** Utilize LLMs *on-premise* ou APIs seguras, e anonimize dados confidenciais. **5. Erros de Formato:** Não especificar o formato de saída pode resultar em dados desorganizados e difíceis de processar. **Contramedida:** Exija formatos estruturados como JSON ou tabelas Markdown.

## URL
[https://www.glean.com/blog/30-ai-prompts-for-finance-professionals](https://www.glean.com/blog/30-ai-prompts-for-finance-professionals)
