# Prompts para Análise Financeira (Financial Analysis Prompts)

## Description

**Prompts para Análise Financeira** são instruções de linguagem natural projetadas para interagir com Large Language Models (LLMs) e agentes de IA, direcionando-os a executar tarefas complexas de finanças, contabilidade e planejamento. O foco é transformar dados brutos de sistemas ERP, planilhas e relatórios em *insights* acionáveis, relatórios executivos e projeções financeiras em tempo real. Esta técnica de engenharia de prompt é crucial para automatizar a Análise e Planejamento Financeiro (FP&A), otimizar o fechamento mensal, gerenciar o fluxo de caixa e garantir a conformidade e a preparação para auditorias [1]. A precisão e a eficácia dependem da clareza do prompt, da especificação do papel (ex: "Como analista financeiro...") e da capacidade do LLM de se integrar a fontes de dados empresariais [2].

## Statistics

- **Acurácia Preditiva:** Em estudos, o GPT-4 demonstrou uma acurácia de 60% na previsão da direção de lucros futuros, superando a acurácia de 53% de analistas humanos [3].
- **Economia de Tempo:** Empresas financeiras de ponta relatam uma economia de tempo de até 95% em tarefas de análise de dados ao usar LLMs com dados verificados [4].
- **Crescimento de Mercado:** O mercado global de Engenharia de Prompt foi avaliado em US$ 222 milhões em 2023 e projetado para atingir US$ 2,06 bilhões até 2030 (CAGR de 32,8%) [5].
- **Eficácia do Prompt:** 83,7% dos entrevistados concordam que prompts mais claros e específicos levam a melhores resultados de IA [6].

## Features

- **Automação de FP&A:** Geração instantânea de previsões, análise de variação orçamentária e modelagem de cenários (ex: impacto de cortes de custos) [1].
- **Fechamento e Contabilidade:** Identificação de transações ausentes, classificação incorreta de fornecedores e análise de fluxo (flux analysis) para contas do Razão Geral (GL) [1].
- **Gestão de Tesouraria:** Visão em tempo real da posição de caixa por entidade e re-previsão de liquidez de curto prazo com base em atividades recentes de Contas a Pagar (AP) e a Receber (AR) [1].
- **Conformidade e Auditoria:** Sinalização de entradas de diário sem documentação e geração de narrativas de variação para notas de auditoria [1].
- **Análise Preditiva:** Capacidade de prever a direção de lucros futuros com maior precisão do que analistas humanos em alguns estudos [3].

## Use Cases

- **FP&A e Orçamento:** Previsão de receita, análise de variação de Despesas Gerais e Administrativas (SG&A) e modelagem de extensão de *runway* [1].
- **Gestão de Capital de Giro:** Otimização de Contas a Receber (AR) e Contas a Pagar (AP), incluindo a identificação de faturas de alto valor em risco de atraso [1].
- **Relatórios Executivos:** Geração de resumos de *burn multiple* e margem operacional para comunicação com investidores e *board* [1].
- **Due Diligence e M&A:** Análise rápida de demonstrações financeiras de empresas-alvo e identificação de riscos contábeis [2].
- **Análise de Investimentos:** Geração de fatores financeiros preditivos e análise de tendências de mercado [3].

## Integration

**Melhores Práticas de Prompting para Finanças:**
1.  **Definir o Papel (Role-Playing):** Comece o prompt estabelecendo o contexto e o papel que a IA deve assumir (ex: "Aja como um CFO...").
2.  **Especificar a Fonte de Dados:** Indique claramente quais dados (ERP, NetSuite, planilhas) o agente deve usar.
3.  **Definir o Formato de Saída:** Peça o resultado em um formato estruturado (tabela, resumo executivo, gráfico) [1].

**Exemplos de Prompts Concretos:**

| Categoria | Prompt de Exemplo | Objetivo |
| :--- | :--- | :--- |
| **FP&A** | "Compare nossa receita mensal e tendências de gastos com marketing de 2025 com *benchmarks* do setor e gere um resumo executivo." [1] | Análise de Eficiência de CAC e Crescimento para *Board* e Investidores. |
| **Tesouraria** | "Qual é a nossa posição total de caixa por entidade, convertida para USD, a partir desta manhã?" [1] | Visibilidade em tempo real da liquidez. |
| **Contabilidade** | "Sinalize quaisquer contas do Razão Geral com variação >10% em relação ao mês passado e explique os fatores." [1] | Automação da Análise de Fluxo (Flux Analysis) para o fechamento. |
| **Análise de Risco** | "Liste os clientes com tendências de pagamento em declínio nos últimos 60 dias e atribua uma classificação de risco." [1] | Identificação precoce de potenciais inadimplências em AR. |
| **Modelagem** | "Qual é o impacto no fluxo de caixa se pausarmos a contratação de G&A até o final do ano? Projete a extensão da pista (runway)." [1] | Modelagem de Cenários Estratégicos. |

## URL

https://www.concourse.co/insights/ai-prompts-for-finance-teams