# Hard Prompting

## Description

Prompts Rígidos (Hard Prompts) são entradas textuais explícitas, legíveis por humanos, escritas em linguagem natural para guiar o comportamento de Modelos de Linguagem Grande (LLMs). Diferentemente dos Soft Prompts, que são vetores de *embedding* otimizados e não interpretáveis, os Hard Prompts são compostos por palavras e tokens interpretáveis, sendo tipicamente criados manualmente por humanos. Eles dependem da criatividade e do conhecimento de domínio do engenheiro de prompt para definir a tarefa e utilizar as capacidades pré-treinadas do modelo. A pesquisa recente foca em métodos de otimização discreta baseada em gradiente para automatizar a descoberta de prompts rígidos de alto desempenho, mantendo sua interpretabilidade e transferibilidade.

## Statistics

O artigo "Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery" (NeurIPS 2023) foi citado mais de **400 vezes**, destacando a relevância da otimização de prompts rígidos. Um estudo de 2023 mostrou que, em algumas tarefas, os Hard Prompts superaram os Soft Prompts em métricas como ROUGE, com o modelo falcon-7b obtendo o melhor resultado. O método de otimização discreta baseado em gradiente é apresentado como uma forma de automatizar a descoberta de prompts rígidos de alto desempenho, que antes dependia de tentativa e erro ou intuição.

## Features

*   **Interpretabilidade e Transparência:** Facilmente compreendido e editado por humanos.
*   **Generalizável e Transferível:** Adaptável em vários domínios e pode ser transferido para diferentes modelos.
*   **Criação Manual:** Exige elaboração e refinamento manual, frequentemente por tentativa e erro.
*   **Otimização Discreta:** Pesquisas recentes (NeurIPS 2023) introduziram a otimização discreta baseada em gradiente para automatizar a descoberta de prompts rígidos de alto desempenho.

## Use Cases

*   **Tarefas de Geração Geral:** Ideal para tarefas que exigem instruções diretas e interpretabilidade, como sumarização, tradução e classificação.
*   **Desenvolvimento de Código:** Geração de trechos de código ou scripts com base em requisitos explícitos.
*   **Criação de Conteúdo:** Geração de textos criativos, artigos ou respostas de suporte ao cliente.
*   **Validação de Conceito:** Recomendado para iniciar o desenvolvimento de aplicações com LLMs, devido à sua clareza e facilidade de depuração.

## Integration

**Exemplos de Prompt:**
1.  **Suporte ao Cliente:** "Resuma a reclamação do cliente e sugira duas soluções."
2.  **Programação:** "Escreva código Python para ordenar um array usando o algoritmo bubble sort."
3.  **Escrita Criativa:** "Gere um poema sobre o outono."

**Melhores Práticas:**
*   **Clareza e Especificidade:** Use linguagem clara e instruções diretas para definir a tarefa.
*   **Modelos de Prompt (Templates):** Crie estruturas reutilizáveis para garantir saídas consistentes em tarefas semelhantes.
*   **Refinamento Iterativo:** Utilize a técnica de Tentativa e Erro para refinar a fraseologia do prompt e otimizar o desempenho.

## URL

https://proceedings.neurips.cc/paper_files/paper/2023/file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf