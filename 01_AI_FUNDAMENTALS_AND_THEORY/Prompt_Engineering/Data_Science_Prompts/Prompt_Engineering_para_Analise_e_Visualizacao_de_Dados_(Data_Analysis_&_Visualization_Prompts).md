# Prompt Engineering para Análise e Visualização de Dados (Data Analysis & Visualization Prompts)

## Description

A Engenharia de Prompt para Análise e Visualização de Dados é a prática de estruturar instruções para Modelos de Linguagem Grande (LLMs) para otimizar o ciclo de vida da Ciência de Dados. Isso inclui desde o planejamento do projeto, passando pela limpeza e Análise Exploratória de Dados (EDA), até a geração de código e o design de visualizações e dashboards. O foco está em transformar a capacidade generativa dos LLMs em ferramentas precisas e contextuais para a manipulação e interpretação de dados. A técnica "Clarify – Confirm – Complete" é um framework chave para garantir que o LLM entenda o contexto e as restrições antes de gerar um plano de análise ou código.

## Statistics

As métricas de desempenho para LLMs em tarefas de Análise de Dados e Visualização são focadas na qualidade e precisão da saída, em vez de taxas de sucesso consolidadas de modelos específicos. As métricas de avaliação recomendadas incluem:
- **Precisão da Resposta (Answer Correctness):** Essencial para garantir que os insights e o código gerado estejam factualmente corretos.
- **Similaridade Semântica (Semantic Similarity):** Usada para avaliar o quão próximo o resultado gerado (e.g., uma explicação de insight) está de uma resposta ideal.
- **Alucinação (Hallucination):** Mede a taxa de informações incorretas ou inventadas, crucial para a confiabilidade do código e das conclusões.
- **Taxa de Sucesso na Geração de Código (Code Generation Success Rate):** A porcentagem de código Python/Pandas/Matplotlib gerado que é executável e produz o resultado esperado.
- **Tempo para Insight (Time-to-Insight):** Métrica de caso de uso que avalia a eficiência do LLM em acelerar o ciclo de vida da Ciência de Dados.

**Citação:** As melhores práticas de avaliação de LLMs são amplamente discutidas em artigos de pesquisa e guias de empresas como Microsoft, DataCamp e Confident AI (referências de pesquisa de 2024-2025).

## Features

**Técnicas de Prompting Específicas:**
1.  **Clarify – Confirm – Complete:** Framework para planejamento de projetos de Data Science, forçando o LLM a refinar o escopo antes de gerar o plano.
2.  **Prompting Baseado em Papel (Role-Based Prompting):** Atribuir ao LLM o papel de "Cientista de Dados Sênior" ou "Especialista em Visualização" para obter respostas mais especializadas.
3.  **Prompting Estruturado para Visualização:** Uso de prompts para sugerir tipos de gráficos para KPIs, design de layout de dashboard, paletas de cores acessíveis e anotações contextuais.
4.  **Geração de Código Boilerplate:** Criação rápida de código Python (Pandas, Matplotlib, Seaborn) para tarefas repetitivas de limpeza e EDA.

**Recursos Chave:**
- **Templates de Prompt para Visualização:** Estruturas para selecionar o tipo de visualização mais adequado para métricas específicas (e.g., linha para tendência, funil para conversão).
- **Templates de Prompt para Dashboard:** Modelos para propor layouts que otimizam a narrativa de dados e reduzem a carga cognitiva.

## Use Cases

1.  **Planejamento de Projetos de Data Science:** Criação de planos de projeto detalhados, incluindo etapas de pré-processamento, engenharia de recursos e seleção de modelos.
2.  **Análise Exploratória de Dados (EDA) Acelerada:** Geração de código Python para tarefas de EDA, como análise de distribuição, correlação e tratamento de valores ausentes.
3.  **Design de Dashboards Estratégicos:** Auxílio na seleção de tipos de visualização para KPIs, design de layout e sugestão de filtros e segmentos para dashboards interativos.
4.  **Storytelling com Dados:** Geração de anotações e insights contextuais para transformar gráficos em narrativas claras para stakeholders não técnicos.
5.  **Otimização de Visualização:** Teste A/B de diferentes estilos de visualização e recomendação de paletas de cores acessíveis.

## Integration

**Exemplos de Prompts e Melhores Práticas:**

| Categoria | Exemplo de Prompt (Inglês) | Exemplo de Prompt (Português) | Melhor Prática |
| :--- | :--- | :--- | :--- |
| **Planejamento de Projeto (Clarify – Confirm – Complete)** | "You are a senior data scientist. I have a dataset on customer churn. Before giving an analysis plan: 1. Clarify what key features are relevant. 2. Confirm the best modeling approach (classification or regression). 3. Then complete a detailed project plan (data cleaning, feature engineering, model options, and reporting steps)." | "Você é um cientista de dados sênior. Tenho um conjunto de dados sobre churn de clientes. Antes de dar um plano de análise: 1. Esclareça quais recursos-chave são relevantes. 2. Confirme a melhor abordagem de modelagem (classificação ou regressão). 3. Em seguida, complete um plano de projeto detalhado (limpeza de dados, engenharia de recursos, opções de modelo e etapas de relatório)." | **Fornecer Contexto e Restrições:** Sempre inclua o papel do LLM, o objetivo final e detalhes sobre o conjunto de dados (tamanho, variáveis, tipo de problema). |
| **Visualização de Dados (Seleção de Tipo)** | "Suggest the best visualization types (e.g., bar, line, heatmap) for the following KPIs: monthly recurring revenue, customer lifetime value, and website conversion rate." | "Sugira os melhores tipos de visualização (por exemplo, barra, linha, mapa de calor) para os seguintes KPIs: receita recorrente mensal, valor vitalício do cliente e taxa de conversão do site." | **Foco no KPI:** Vincule o tipo de visualização diretamente ao KPI para garantir que o gráfico conte a história certa (e.g., linha para tendência, funil para conversão). |
| **Análise Exploratória de Dados (EDA)** | "Generate Python code using Pandas and Matplotlib to perform a correlation analysis between 'age', 'income', and 'purchase_amount' in the provided DataFrame. Also, suggest a visualization to represent the findings." | "Gere código Python usando Pandas e Matplotlib para realizar uma análise de correlação entre 'idade', 'renda' e 'valor_da_compra' no DataFrame fornecido. Além disso, sugira uma visualização para representar as descobertas." | **Geração de Código Específica:** Peça explicitamente o código e a biblioteca (e.g., Pandas, Seaborn) e o resultado desejado (e.g., análise de correlação, histograma). |
| **Design de Dashboard (Layout)** | "Propose a layout design for a dashboard that shows sales performance, focusing on regional comparison and monthly trend analysis. The target audience is executive leadership." | "Proponha um design de layout para um dashboard que mostre o desempenho de vendas, com foco na comparação regional e na análise de tendências mensais. O público-alvo é a liderança executiva." | **Definir Público e Objetivo:** O design do dashboard deve ser adaptado ao público-alvo para otimizar o tempo de insight. |

**Melhores Práticas Adicionais:**
- **Anotações Contextuais:** Use prompts para gerar anotações que transformam visuais em ferramentas de storytelling (e.g., "Gere texto de anotação para o pico de vendas em julho, explicando a causa provável").
- **Teste A/B de Visualização:** Peça ao LLM para projetar duas versões rivais de um gráfico e critérios para testar qual delas oferece o melhor "tempo para insight".

## URL

https://towardsdatascience.com/become-a-better-data-scientist-with-these-prompt-engineering-hacks/