# Prompts de Pesquisa de Investimento Estruturada (RTO Framework)

## Description

A Engenharia de Prompt para Pesquisa de Investimento refere-se à aplicação de técnicas estruturadas para maximizar a precisão e a relevância das respostas de Grandes Modelos de Linguagem (LLMs) em análises financeiras e de mercado. O cerne desta técnica é a estrutura de três pilares: **Função (Role)**, **Tarefa (Task)** e **Saída (Output)**. Ao definir o LLM como um especialista (ex: analista de ações), detalhar a tarefa com métricas e períodos específicos, e formatar a saída desejada (ex: em bullet points com citações diretas), os analistas mitigam significativamente o risco de alucinações e obtêm *insights* acionáveis. A prática é crucial para escalar a análise de mercado, permitindo que os investidores examinem um volume maior de empresas e testem mais hipóteses com qualidade e auditabilidade [1].

## Statistics

**Citações Acadêmicas:** Pesquisas recentes (2023-2025) destacam o impacto da *prompt engineering* na tomada de decisão financeira [3] [5]. O artigo "Review of Prompt Engineering Techniques in Finance" (2025) foi citado por mais de 40 fontes, indicando a relevância crescente do tema [2]. **Métricas de Desempenho:** Artigos de pesquisa comparam o desempenho de LLMs (incluindo GPT-5, Gemini 2.5 Pro, Claude Opus) em tarefas de raciocínio financeiro, com análises que envolvem mais de 60 execuções para medir a eficácia de diferentes *prompts* [6]. **Escalabilidade:** O uso de *prompts* estruturados permite uma análise de "80% de qualidade em mil empresas, em vez de 99% em 10 empresas," aumentando drasticamente a escalabilidade da pesquisa [1].

## Features

**Estrutura de Três Pilares (RTO):** Define a **Função** (Role) do LLM (ex: analista de crédito), a **Tarefa** (Task) a ser executada (ex: análise YoY de receita) e o formato de **Saída** (Output) desejado (ex: tabela, resumo executivo). **Técnicas Avançadas:** Integração de métodos de raciocínio complexo como *Chain-of-Thought* (CoT), *Tree-of-Thought* (ToT) e *Graph-of-Thought* (GoT) para melhorar a capacidade de raciocínio financeiro dos modelos [2] [3]. **Auditabilidade:** Exigência de fontes auditáveis e citações diretas para mitigar alucinações, um ponto crítico na análise financeira. **Frameworks de Biblioteca:** Utilização de repositórios centralizados para armazenar, gerenciar e compartilhar *templates* de *prompts* estruturados e testados [4].

## Use Cases

**Análise de Demonstrações Financeiras:** Resumo de dados, análise de tendências, rascunho de notas de divulgação e identificação de anomalias em relatórios financeiros (Balanço, DRE, Fluxo de Caixa). **Análise de Portfólio:** Avaliação de risco e retorno, e sugestão de rebalanceamento de ativos. **Due Diligence e Pesquisa de Mercado:** Automação da triagem de propostas, pesquisa de mercado e elaboração de memorandos de investimento. **Teste de Hipóteses:** Uso de LLMs para determinar a direção dos lucros futuros e o impacto de eventos macroeconômicos em setores específicos. **Prospecção:** Geração de *prompts* para planejamento, investimento e prospecção de clientes para consultores financeiros [4].

## Integration

A integração eficaz requer a adoção da estrutura RTO (Role, Task, Output) e a utilização de *prompts* específicos para cada tipo de análise.

**Melhores Práticas:**
1.  **Definir a Função (Role):** Sempre comece o *prompt* definindo o papel do LLM (ex: "Aja como um analista de *equity* especializado em bancos de investimento...").
2.  **Especificar a Tarefa (Task):** Detalhe o período (Q4 2024), as métricas (receita de *investment banking*, QoQ) e o contexto (comparação YoY) [1].
3.  **Exigir Formato (Output):** Peça a saída em um formato consumível (ex: "Comece com o desempenho do Q4, seguido por tendências YoY/sequenciais, e então a perspectiva da gestão. Cite a gestão diretamente ao discutir a perspectiva.").

**Exemplos de Prompts:**
*   **Análise de Demonstrações Financeiras:** "Como analista financeiro, analise o Balanço Patrimonial e a Demonstração do Resultado do Exercício da [Nome da Empresa] para o último trimestre. Identifique as três principais mudanças em relação ao trimestre anterior e explique o impacto potencial no fluxo de caixa futuro."
*   **Análise de Portfólio:** "Avalie o risco e o retorno do meu portfólio de investimentos com base nas condições atuais do mercado, sugerindo uma alocação de ativos mais defensiva e justificando a mudança com base em [Evento Macroeconômico]."
*   **Análise de Tendências:** "Analise o impacto da [Regulamentação Governamental] no setor de [Setor] e identifique três empresas que estão mais bem posicionadas para se beneficiar, fornecendo uma breve justificativa para cada uma."

## URL

https://www.ai-street.co/p/effective-prompts-for-investment-research