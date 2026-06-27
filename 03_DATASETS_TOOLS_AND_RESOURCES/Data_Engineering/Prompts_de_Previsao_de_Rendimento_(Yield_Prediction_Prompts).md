# Prompts de Previsão de Rendimento (Yield Prediction Prompts)

## Description
Prompts de Previsão de Rendimento são instruções estruturadas fornecidas a Modelos de Linguagem Grande (LLMs) ou outros modelos de IA para solicitar a previsão de um resultado futuro (o "rendimento" ou "yield") com base em dados históricos, contextuais e regras específicas. Diferentemente de prompts de geração de texto simples, estes prompts exigem que o modelo realize uma análise de dados, identifique padrões e aplique lógica de *forecasting* para estimar métricas como lucro, produção agrícola, desempenho de mercado ou resultados de processos industriais. A eficácia desses prompts depende criticamente da clareza da solicitação, da inclusão de dados relevantes (seja diretamente no prompt ou por meio de ferramentas/contexto) e, em alguns casos, da solicitação de um raciocínio passo a passo (Chain-of-Thought) para justificar a previsão. Pesquisas recentes (2025) sugerem que, para tarefas complexas como previsão, o refinamento básico do prompt tem ganhos limitados, mas a inclusão de referências a taxas base (*base rates*) pode trazer benefícios. O foco principal é converter a capacidade de raciocínio do LLM em uma ferramenta de análise preditiva.

## Examples
```
**1. Previsão de Colheita Agrícola (Agricultura):**
"Atue como um agrônomo. Preveja o rendimento (toneladas por hectare) de milho para a próxima safra, considerando os seguintes dados:
- Histórico de 3 anos: [2.5, 2.8, 2.6] t/ha
- Chuva acumulada (últimos 3 meses): 350mm (Média histórica: 400mm)
- Temperatura média: 25°C (Ideal: 24°C)
- Tipo de solo: Argiloso (Alto teor de nutrientes)
- Doenças/Pragas: Presença leve de cigarrinha.
Forneça a previsão e o raciocínio detalhado."

**2. Previsão de Vendas (Negócios/Finanças):**
"Você é um analista de vendas. Preveja a receita trimestral (Q4) para o produto 'Software X'.
- Receita Q1, Q2, Q3: [$1.2M, $1.5M, $1.8M]
- Lançamento de concorrente: Q3 (impacto estimado de -10% nas vendas)
- Campanha de marketing: Lançamento em Q4 (aumento estimado de +20% nas vendas).
Calcule a previsão de receita e justifique as ponderações de impacto."

**3. Previsão de Desempenho de Processo (Indústria/Tecnologia):**
"Como engenheiro de produção, preveja a taxa de defeito (Yield Rate) da linha de montagem de semicondutores para a próxima semana.
- Taxa de defeito média (últimas 4 semanas): [3.2%, 3.0%, 3.5%, 3.1%]
- Manutenção programada: Sim, no início da semana (redução esperada de 0.5% na taxa de defeito).
- Novo lote de matéria-prima: Qualidade inferior (aumento esperado de 1.0% na taxa de defeito).
Qual é a taxa de defeito prevista? Apresente o cálculo passo a passo."

**4. Previsão de Retorno de Investimento (Finanças):**
"Analise o seguinte investimento em marketing digital:
- Investimento inicial: $50.000
- Custo por Aquisição (CPA) histórico: $50
- Taxa de Conversão (CVR) esperada: 5%
- Valor Médio de Pedido (AOV): $200
Preveja o Retorno sobre o Investimento (ROI) após 6 meses, assumindo 1.000 cliques por mês. Apresente o ROI em porcentagem e o número de clientes esperados."

**5. Previsão de Tráfego Web (Tecnologia):**
"Preveja o número de usuários ativos diários (DAU) para o próximo mês, com base nos seguintes dados:
- DAU médio (últimos 3 meses): [10.000, 12.000, 15.000]
- Evento de lançamento: Um grande recurso será lançado na metade do mês (aumento esperado de 30% no DAU).
- Sazonalidade: Férias escolares no final do mês (redução esperada de 15% no DAU).
Qual é a previsão de DAU para o próximo mês? Use o método de extrapolação e ajuste."
```

## Best Practices
**1. Estrutura e Contexto:** Sempre defina o papel do modelo (ex: "Você é um analista financeiro experiente") e forneça o máximo de contexto e dados históricos possível. **2. Dados Estruturados:** Inclua os dados de entrada em formato estruturado (tabelas, JSON, CSV) para facilitar a análise do modelo. **3. Raciocínio Explícito (CoT):** Peça ao modelo para detalhar o processo de raciocínio (Chain-of-Thought) antes de apresentar a previsão final. Isso aumenta a transparência e a precisão. **4. Limites e Variáveis:** Especifique o horizonte de tempo da previsão, as métricas de saída desejadas e quaisquer variáveis ou suposições que o modelo deve considerar. **5. Cenários:** Solicite previsões em múltiplos cenários (otimista, pessimista, base) para uma análise de risco mais completa.

## Use Cases
nan

## Pitfalls
**1. Falta de Contexto:** Fornecer apenas dados brutos sem especificar o papel do modelo, o objetivo da previsão ou o horizonte de tempo. **2. Dados Insuficientes ou Irrelevantes:** Esperar uma previsão precisa com pouquíssimos pontos de dados ou incluir variáveis que não têm impacto comprovado no resultado. **3. Viés de Confirmação:** Formular o prompt de forma a guiar o modelo para uma resposta desejada, em vez de permitir uma análise objetiva. **4. Ignorar *Base Rates*:** Não fornecer ou solicitar que o modelo considere as taxas base ou a frequência histórica do evento, o que é crucial para previsões mais robustas. **5. Ambiguidade na Métrica:** Não definir claramente a métrica de "rendimento" (ex: lucro bruto vs. lucro líquido, produção total vs. produção por área).

## URL
[https://arxiv.org/abs/2506.01578](https://arxiv.org/abs/2506.01578)
