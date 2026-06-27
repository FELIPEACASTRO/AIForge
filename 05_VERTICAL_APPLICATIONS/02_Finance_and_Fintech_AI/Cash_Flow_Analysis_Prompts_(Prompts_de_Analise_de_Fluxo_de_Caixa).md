# Cash Flow Analysis Prompts (Prompts de Análise de Fluxo de Caixa)

## Description
Prompts de Análise de Fluxo de Caixa são instruções estruturadas fornecidas a Modelos de Linguagem Grande (LLMs) para processar dados financeiros de entradas e saídas de caixa. Eles transformam a IA em um assistente financeiro capaz de analisar tendências, identificar riscos, criar orçamentos e prever o fluxo de caixa futuro. A eficácia reside na capacidade do prompt de fornecer contexto (função da IA, dados, objetivo) e diretrizes de resposta (formato, métricas, recomendações) para obter *insights* financeiros acionáveis e precisos. Essa técnica é fundamental para a gestão financeira, permitindo que profissionais e empresários tomem decisões proativas para otimizar a liquidez e a solvência de uma organização.

## Examples
```
**1. Análise Detalhada de Fluxo de Caixa:**
`#CONTEXTO: Você é um analista financeiro sênior. #DADOS: [Insira dados de entradas e saídas de caixa dos últimos 6 meses, categorizados por fonte/destino]. #OBJETIVO: Fornecer uma análise concisa que destaque padrões, anomalias e os 3 principais fatores de variação do fluxo de caixa. #DIRETRIZES: 1. Calcule a Margem de Fluxo de Caixa. 2. Identifique o mês com a maior e a menor liquidez. 3. Sugira uma ação imediata para melhorar o saldo final.`

**2. Previsão de Fluxo de Caixa para 90 Dias:**
`#CONTEXTO: Você é um especialista em modelagem financeira. #DADOS: [Insira dados históricos de 12 meses e premissas de crescimento de vendas (5%) e aumento de custos (2%)]. #OBJETIVO: Criar uma projeção de fluxo de caixa para os próximos 3 meses. #DIRETRIZES: 1. Apresente os resultados em formato de tabela (Mês 1, Mês 2, Mês 3). 2. Destaque qualquer mês com saldo de caixa negativo projetado. 3. Explique a metodologia de previsão utilizada.`

**3. Identificação de Riscos de Liquidez:**
`#CONTEXTO: Você é um auditor de risco financeiro. #DADOS: [Insira o ciclo de conversão de caixa atual (ex: 45 dias) e o prazo médio de recebimento (ex: 60 dias)]. #OBJETIVO: Identificar vulnerabilidades de liquidez e propor estratégias de mitigação. #DIRETRIZES: 1. Liste 3 riscos críticos (ex: dependência de um único cliente, sazonalidade). 2. Para cada risco, forneça uma estratégia de mitigação acionável (ex: negociar prazos de pagamento com fornecedores).`

**4. Otimização de Capital de Giro:**
`#CONTEXTO: Você é um consultor de eficiência operacional. #DADOS: [Insira o valor atual do Capital de Giro e o saldo de caixa mínimo desejado]. #OBJETIVO: Sugerir 5 maneiras de otimizar o uso do capital de giro para liberar caixa. #DIRETRIZES: As sugestões devem focar em (a) gestão de estoque, (b) contas a receber e (c) contas a pagar. Apresente em formato de lista numerada.`

**5. Análise de Cenários (Otimista vs. Pessimista):**
`#CONTEXTO: Você é um estrategista de negócios. #DADOS: [Insira o fluxo de caixa base do último trimestre]. #OBJETIVO: Simular o impacto no saldo final de caixa sob dois cenários: Otimista (aumento de 15% nas vendas) e Pessimista (queda de 10% nas vendas e atraso de 30 dias nos recebimentos). #DIRETRIZES: Compare os saldos finais de caixa dos três cenários (Base, Otimista, Pessimista) e forneça uma conclusão sobre a resiliência financeira da empresa.`

**6. Criação de Orçamento de Caixa:**
`#CONTEXTO: Você é um planejador orçamentário. #DADOS: [Insira as categorias de despesas fixas e variáveis e as fontes de receita]. #OBJETIVO: Elaborar um orçamento de caixa mensal detalhado. #DIRETRIZES: 1. Separe receitas e despesas. 2. Inclua uma coluna para a variação orçamentária (Orçado vs. Realizado). 3. O orçamento deve ser facilmente exportável para uma planilha.`
```

## Best Practices
**1. Estrutura e Contexto:** Sempre defina o papel da IA (ex: "analista financeiro sênior"), forneça o contexto (dados brutos ou resumidos) e estabeleça o objetivo claro (ex: "identificar os 3 principais riscos"). **2. Diretrizes de Resposta:** Use a seção `#RESPONSE GUIDELINES` para especificar o formato de saída (tabela, relatório, lista), as métricas a serem calculadas (margem de fluxo de caixa, ciclo de conversão de caixa) e o tipo de recomendação esperada (estratégias de mitigação, ajustes orçamentários). **3. Itere e Refine:** Comece com prompts amplos e use prompts de acompanhamento para aprofundar a análise (ex: "Com base na previsão, quais são as opções de financiamento de curto prazo?"). **4. Confidencialidade:** Nunca insira dados confidenciais ou de identificação pessoal. Use dados anonimizados ou simulados.

## Use Cases
**1. Planejamento Orçamentário:** Criação de orçamentos de caixa detalhados e projeções financeiras de curto e longo prazo. **2. Gestão de Risco:** Identificação proativa de períodos de baixa liquidez ou riscos de insolvência. **3. Otimização de Capital de Giro:** Sugestões para acelerar recebimentos (contas a receber) e gerenciar pagamentos (contas a pagar) para liberar caixa. **4. Análise de Desempenho:** Comparação do fluxo de caixa real com o orçado, identificando desvios e suas causas. **5. Tomada de Decisão Estratégica:** Avaliação do impacto financeiro de investimentos, aquisições ou cortes de custos antes de sua implementação. **6. Relatórios e Comunicação:** Geração de resumos executivos e relatórios de fluxo de caixa para *stakeholders* e conselhos de administração.

## Pitfalls
**1. Inserção de Dados Brutos Não Estruturados:** Fornecer dados financeiros em um formato de texto longo e desorganizado (em vez de tabelas ou listas com rótulos claros) leva a erros de interpretação e resultados imprecisos. **2. Falta de Contexto:** Não definir o papel da IA ou o objetivo da análise resulta em respostas genéricas e pouco úteis. A IA precisa saber se deve agir como um contador, um analista de risco ou um estrategista. **3. Ignorar Premissas:** Não incluir premissas de negócios (ex: sazonalidade, lançamento de novos produtos, inflação) na previsão de fluxo de caixa resulta em projeções irrealistas. **4. Confiança Excessiva:** Tratar a saída da IA como um fato consumado sem validação humana. A IA é uma ferramenta de análise, não um substituto para a *due diligence* financeira. **5. Prompts Únicos e Longos:** Tentar resolver toda a análise em um único prompt complexo. É mais eficaz usar uma série de prompts de acompanhamento para refinar a análise e aprofundar os *insights*.

## URL
[https://www.godofprompt.ai/blog/prompts-to-improve-your-cash-flow](https://www.godofprompt.ai/blog/prompts-to-improve-your-cash-flow)
