# Fundamental Analysis Prompts

## Description
**Prompts de Análise Fundamentalista** são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) com o objetivo de realizar uma avaliação aprofundada da saúde financeira, desempenho operacional e valor intrínseco de uma empresa ou ativo. Diferentemente de prompts genéricos, eles exigem que a IA adote uma **persona especializada** (ex: analista de ações, contador, gestor de portfólio) e processe dados financeiros específicos (como relatórios 10-K, demonstrações de resultados, balanços patrimoniais e notícias de mercado) para calcular métricas, identificar tendências, realizar comparações setoriais e formular conclusões de investimento. A eficácia desses prompts reside na sua capacidade de mitigar a tendência da IA a "alucinar" dados, exigindo fontes verificáveis e um raciocínio lógico e transparente, transformando o LLM de um gerador de texto em um assistente de análise quantitativa e qualitativa [1] [2]. A subcategoria mais apropriada é **Finance**.

## Examples
```
**Exemplo 1: Análise de Liquidez e Solvência**
```
**Role:** Atue como um analista de crédito sênior.
**Task:** Analise a liquidez e solvência da "Empresa X" com base nos dados do Balanço Patrimonial de 2023 e 2024.
1. Calcule o Índice de Liquidez Corrente e o Índice de Endividamento Total para ambos os anos.
2. Identifique a variação percentual YoY em cada índice.
3. Compare os resultados com a média do setor (Índice de Liquidez Corrente: 1.5x; Endividamento Total: 0.6x).
**Output:** Apresente os cálculos em uma tabela e forneça um parecer de 4 frases sobre a saúde financeira da empresa.
```

**Exemplo 2: Avaliação de Rentabilidade e Eficiência**
```
**Role:** Você é um gestor de portfólio focado em valor.
**Task:** Avalie a rentabilidade e a eficiência operacional da "Empresa Y" no último trimestre (Q3 2024).
1. Calcule a Margem Bruta, Margem Operacional e Retorno sobre o Patrimônio Líquido (ROE).
2. Explique as principais variações (acima de 5%) nos custos operacionais em relação ao trimestre anterior.
3. Determine se a empresa está gerando valor para os acionistas.
**Output:** Responda em formato de relatório executivo, destacando as métricas-chave em negrito e citando as fontes de dados.
```

**Exemplo 3: Análise de Fluxo de Caixa e Investimento**
```
**Role:** Analista de Fusões e Aquisições (M&A).
**Task:** Analise o Fluxo de Caixa Livre (FCF) da "Empresa Z" nos últimos 5 anos.
1. Calcule o FCF e o FCF por Ação.
2. Avalie a sustentabilidade dos gastos de capital (CAPEX) em relação à depreciação.
3. Projete o FCF para o próximo ano, assumindo um crescimento de receita de 8% e margens estáveis.
**Output:** Apresente os dados históricos em uma lista e a projeção em um parágrafo, com uma nota sobre a qualidade do FCF.
```

**Exemplo 4: Análise Qualitativa de Vantagem Competitiva**
```
**Role:** Estrategista de mercado.
**Task:** Realize uma análise qualitativa da vantagem competitiva (Moat) da "Empresa Alpha" no setor de SaaS.
1. Aplique o framework das Cinco Forças de Porter para avaliar a atratividade do setor.
2. Identifique e descreva o tipo de Moat (ex: Efeitos de Rede, Economias de Escala, Ativos Intangíveis).
3. Conclua se o Moat é durável e defensável.
**Output:** Estruture a resposta com títulos e subtítulos claros para cada seção da análise.
```

**Exemplo 5: Análise de Relatório de Earnings Call**
```
**Role:** Analista de Relações com Investidores.
**Task:** Revise a transcrição da última teleconferência de resultados (Earnings Call) da "Empresa Beta".
1. Extraia e liste todas as menções a "crescimento de margem" e "desafios regulatórios".
2. Resuma o tom geral do CEO (otimista, cauteloso, neutro).
3. Identifique 3 perguntas-chave feitas pelos analistas e as respostas da gestão.
**Output:** Use bullet points para as listas e um parágrafo para o resumo do tom.
```

**Exemplo 6: Análise de Cenário e Sensibilidade**
```
**Role:** Consultor de risco financeiro.
**Task:** Realize uma análise de sensibilidade para o Lucro por Ação (LPA) da "Empresa Gama".
1. Calcule o LPA atual.
2. Modele o LPA em três cenários: (A) Aumento de 10% no custo da matéria-prima, (B) Queda de 5% no volume de vendas, (C) Combinação de (A) e (B).
3. Apresente o impacto percentual no LPA para cada cenário.
**Output:** Tabela comparativa dos cenários e uma conclusão sobre a resiliência do LPA.
```
```

## Best Practices
**1. Defina o Papel e o Contexto (Role and Context):** Comece o prompt instruindo a IA a agir como um analista financeiro, de crédito ou de ações, especificando o setor e o mercado (ex: "Aja como um analista de ações especializado em tecnologia de semicondutores").
**2. Estrutura de Prompt Detalhada (Detailed Prompt Structure):** Utilize a estrutura **Role (Papel), Task (Tarefa), Output (Saída)**. A tarefa deve ser dividida em sub-tarefas claras (ex: "1. Analisar o crescimento da receita YoY. 2. Calcular o índice Dívida/EBITDA. 3. Comparar com a média do setor").
**3. Forneça Dados de Entrada (Provide Input Data):** Sempre que possível, inclua os dados brutos ou o link para a fonte (ex: "Com base nos seguintes dados do 10-K de 2023: [dados/link]"). A IA não deve "adivinhar" os números.
**4. Exija Raciocínio em Cadeia (Chain-of-Thought):** Peça à IA para mostrar os passos de cálculo e o raciocínio por trás da conclusão (ex: "Explique o processo de cálculo do Fluxo de Caixa Livre antes de apresentar o resultado final").
**5. Validação e Auditoria (Validation and Auditability):** Peça referências e notas de rodapé para cada dado ou afirmação (ex: "Para cada métrica financeira, cite a seção e a página do relatório de onde o dado foi extraído").
**6. Especificidade na Saída (Output Specificity):** Defina o formato de saída (tabela, bullet points, parágrafo), o nível de detalhe e o tom (ex: "Apresente os resultados em uma tabela Markdown, com um resumo executivo de 3 parágrafos no final").

## Use Cases
**1. Due Diligence e Avaliação de Empresas (Due Diligence and Company Valuation):** Automatizar a extração de métricas-chave (P/L, EV/EBITDA, Margens) de relatórios financeiros para acelerar o processo de *due diligence* em fusões e aquisições (M&A) ou investimentos de capital de risco.
**2. Análise de Relatórios de Resultados (Earnings Report Analysis):** Processar transcrições de teleconferências de resultados (*earnings calls*) para resumir o sentimento da gestão, identificar riscos e oportunidades, e extrair as principais perguntas dos analistas.
**3. Comparação Setorial (Sector Comparison):** Realizar análises comparativas de múltiplos e índices financeiros entre empresas concorrentes em um setor específico, identificando *outliers* e líderes de mercado.
**4. Modelagem de Cenários e Testes de Estresse (Scenario Modeling and Stress Testing):** Criar prompts para simular o impacto de variáveis macroeconômicas (ex: aumento da taxa de juros, inflação) ou microeconômicas (ex: perda de um cliente chave) nas demonstrações financeiras de uma empresa.
**5. Geração de Relatórios e Memos (Report and Memo Generation):** Gerar rascunhos de relatórios de pesquisa de ações, memos de investimento ou seções de relatórios anuais, economizando tempo do analista na redação inicial.
**6. Análise de ESG (Environmental, Social, and Governance):** Extrair e analisar dados não financeiros de relatórios de sustentabilidade para avaliar o impacto de fatores ESG no risco e valor de longo prazo da empresa.

## Pitfalls
**1. Alucinação de Dados Financeiros (Financial Data Hallucination):** O maior risco é a IA inventar números, índices ou datas de relatórios que parecem autênticos, mas são falsos. Isso é especialmente perigoso em finanças, onde a precisão é crítica [3].
**2. Falha em Raciocínio Matemático Complexo (Failure in Complex Math):** LLMs são modelos de linguagem, não calculadoras. Eles podem cometer erros em cálculos complexos, como o cálculo de Fluxo de Caixa Descontado (DCF) ou a agregação de dados de múltiplas fontes [4].
**3. Dependência Excessiva de Dados de Treinamento (Over-reliance on Training Data):** A IA pode basear a análise em dados desatualizados ou no conhecimento geral, ignorando informações críticas e recentes que não foram fornecidas no prompt (ex: um evento regulatório recente ou um relatório trimestral novo) [5].
**4. Viés e Generalização (Bias and Generalization):** A IA pode aplicar vieses implícitos em seus dados de treinamento, ou generalizar demais a partir de um único ponto de dados, falhando em considerar as nuances específicas do setor ou da empresa.
**5. Ambiguidade na Linguagem Financeira (Ambiguity in Financial Language):** Termos como "receita" ou "lucro" podem ter diferentes definições contábeis (ex: IFRS vs. GAAP). A falta de especificação no prompt pode levar a cálculos incorretos ou comparações inválidas.
**6. Ignorar a Fonte de Dados (Ignoring Data Source):** Não especificar a fonte de dados (ex: "Use apenas dados auditados do 10-K") pode levar a IA a misturar dados de fontes não confiáveis (ex: artigos de notícias, fóruns) com dados oficiais.

## URL
[https://www.ai-street.co/p/effective-prompts-for-investment-research](https://www.ai-street.co/p/effective-prompts-for-investment-research)
