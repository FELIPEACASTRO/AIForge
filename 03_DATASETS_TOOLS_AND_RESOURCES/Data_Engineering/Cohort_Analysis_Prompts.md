# Cohort Analysis Prompts

## Description
Prompts de Análise de Coorte são técnicas de *prompt engineering* que visam instruir um Grande Modelo de Linguagem (LLM) a realizar ou auxiliar na **Análise de Coorte**. A Análise de Coorte é um método de análise comportamental que rastreia o comportamento de grupos específicos de usuários (coortes) que compartilham características ou experiências comuns (geralmente baseadas no tempo, como o mês de primeira compra) ao longo do tempo. O uso de prompts permite que o LLM processe dados brutos (como CSVs ou tabelas) para definir coortes, calcular métricas (retenção, *churn*, receita) e gerar visualizações ou interpretações, especialmente útil para usuários não familiarizados com ferramentas de planilhas ou análise de dados tradicional [1] [2]. Embora o LLM possa não ser a ferramenta ideal para a análise de dados complexa em si, ele atua como um **planejador, formatador e intérprete** poderoso, permitindo que o usuário obtenha um "primeiro corte" rápido e compreensível da análise [1] [3].

## Examples
```
1. **Prompt de Geração de Tabela de Coorte (Receita):**
```
Você é um Analista de Receita. Dada a seguinte tabela de transações (CSV), crie uma tabela de Análise de Coorte alinhada à direita. A coorte deve ser definida pelo 'mês de primeira compra' do cliente. As colunas devem representar os meses subsequentes de transação. O valor em cada célula deve ser a soma da 'receita' gerada por aquela coorte naquele mês.

[Insira aqui a tabela CSV de dados de transação: customer_id, purchase_date, revenue]
```

2. **Prompt de Geração de Tabela de Coorte (Retenção Percentual):**
```
Com base nos dados de uso fornecidos (user_id, signup_date, activity_month), gere uma tabela de Análise de Coorte de Retenção. A coorte deve ser definida pelo 'mês de inscrição'. O valor em cada célula deve ser a porcentagem de usuários da coorte original que permaneceram ativos no mês subsequente. O mês de inscrição deve ser 100%.
```

3. **Prompt de Interpretação de Resultados:**
```
Analise a tabela de retenção de coorte que acabei de fornecer. Identifique a coorte com o melhor desempenho e a coorte com o pior desempenho após 3 meses. Explique as possíveis razões para essa diferença e sugira uma hipótese de teste A/B para melhorar a retenção da coorte de pior desempenho.
```

4. **Prompt de Visualização de Dados:**
```
Com base na tabela de Análise de Coorte de Receita que você gerou, crie um código Python (usando Pandas e Matplotlib) para gerar um gráfico de área empilhada (Layer Cake Chart) que visualize a contribuição de receita de cada coorte ao longo do tempo.
```

5. **Prompt de Definição de Coorte Comportamental:**
```
Defina 3 coortes comportamentais para um aplicativo de fitness com base nos seguintes critérios: Coorte A: Usuários que completaram 5 treinos na primeira semana. Coorte B: Usuários que abriram o aplicativo, mas não completaram nenhum treino. Coorte C: Usuários que cancelaram a assinatura após o período de teste. Para cada coorte, descreva 3 métricas-chave de acompanhamento.
```

6. **Prompt de Planejamento de Análise:**
```
Atue como um Planejador de Análise de Produto. Quero realizar uma Análise de Coorte para entender o impacto de um novo recurso ('Modo Escuro') lançado em 1º de Março de 2024. Crie um plano de análise detalhado, incluindo: 1. Definição das coortes (antes e depois do lançamento). 2. Métricas a serem comparadas. 3. Período de análise. 4. Um prompt de acompanhamento para o LLM executar a análise.
```

7. **Prompt de Análise de Churn:**
```
Usando a tabela de coorte de retenção fornecida, calcule a taxa de *churn* (abandono) para cada coorte no 4º mês. Qual coorte apresenta o *churn* mais alto e o mais baixo? Formule uma pergunta de pesquisa para entender a causa raiz do *churn* mais alto.
```
```

## Best Practices
**1. Estrutura de Prompt Clara:** Defina claramente o papel do LLM (ex: "Você é um Analista de Dados"), o formato de entrada dos dados (ex: "A tabela CSV a seguir"), a definição da coorte (ex: "Coorte baseada no mês de primeira compra") e a métrica de saída desejada (ex: "Tabela de retenção mês a mês"). **2. Fornecer Contexto e Metas:** Inclua o objetivo da análise (ex: "Identificar a coorte com a maior retenção após 6 meses") para guiar a interpretação do LLM. **3. Iteração e Refinamento:** Use o LLM para refinar a análise. Peça primeiro a tabela de coorte e, em seguida, use um prompt de acompanhamento para pedir a interpretação ou a visualização (ex: "Gere um gráfico de área empilhada com base na tabela acima"). **4. Validação Humana:** Sempre verifique a plausibilidade dos resultados gerados pelo LLM, pois eles são baseados em um modelo estocástico e não substituem a auditoria de dados [1].

## Use Cases
**1. Análise de Retenção de Clientes (SaaS):** Rastrear a retenção de usuários que se inscreveram em diferentes meses para identificar tendências sazonais ou o impacto de mudanças no produto. **2. Avaliação de Campanhas de Marketing:** Comparar o valor de vida útil (*Lifetime Value* - LTV) de coortes de clientes adquiridos por diferentes canais de marketing (ex: Google Ads vs. Mídias Sociais). **3. Otimização de Produto:** Analisar o comportamento de coortes de usuários que experimentaram um novo recurso para medir seu impacto na retenção e engajamento. **4. Análise de Receita:** Criar gráficos de "bolo em camadas" (*Layer Cake Charts*) para visualizar como as coortes mais antigas e mais recentes contribuem para a receita total ao longo do tempo [1]. **5. Segmentação Comportamental:** Identificar grupos de usuários com padrões de uso semelhantes (coortes comportamentais) para direcionar comunicações ou ofertas específicas.

## Pitfalls
**1. Confiança Excessiva nos Dados Brutos:** O LLM só pode analisar os dados que lhe são fornecidos. Se os dados de entrada (CSV, tabela) estiverem sujos, inconsistentes ou incompletos, a análise de coorte será falha. **2. Limitações de Contexto:** Para grandes conjuntos de dados, o LLM pode atingir o limite de tokens, resultando em truncamento ou análise incompleta. **3. Falta de Plausibilidade:** O LLM pode gerar resultados que parecem corretos, mas são estatisticamente implausíveis ou incorretos. A validação humana é crucial [1]. **4. Complexidade da Visualização:** Embora os LLMs possam gerar código para visualizações, a execução e o refinamento de gráficos complexos (como gráficos de área empilhada) podem exigir várias iterações de prompt e, muitas vezes, são mais eficientes em ferramentas dedicadas [1]. **5. Ambiguidade na Definição de Coorte:** Prompts que não definem claramente o evento de coorte (ex: "primeiro login" vs. "primeira compra") levarão a resultados inconsistentes.

## URL
[https://www.saas.wtf/p/saas-cohort-analysis-using-ai](https://www.saas.wtf/p/saas-cohort-analysis-using-ai)
