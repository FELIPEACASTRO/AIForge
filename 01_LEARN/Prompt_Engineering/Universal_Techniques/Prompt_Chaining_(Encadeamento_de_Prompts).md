# Prompt Chaining (Encadeamento de Prompts)

## Description

**Prompt Chaining** (Encadeamento de Prompts) é uma técnica avançada de engenharia de prompt que consiste em dividir uma tarefa complexa em uma sequência de subtarefas menores e mais gerenciáveis. O resultado (output) de um Modelo de Linguagem Grande (LLM) em uma etapa é usado como a entrada (input) para o LLM na etapa seguinte, criando um fluxo de trabalho modular e lógico [1] [2].

Essa abordagem transforma LLMs de meras ferramentas de resposta a perguntas em componentes de um pipeline de processamento de dados mais sofisticado. Ao modularizar o processo, o Prompt Chaining melhora a **confiabilidade**, a **mensurabilidade** e a **escalabilidade** dos sistemas baseados em LLM, permitindo que eles lidem com tarefas que excederiam o escopo de um único prompt detalhado [3] [4].

Existem diversos tipos de encadeamento, incluindo:
*   **Encadeamento Sequencial:** O tipo mais simples, onde a saída de um prompt é diretamente passada para o próximo. Ideal para tarefas com progressão linear.
*   **Encadeamento de Ramificação (Branching):** Uma única saída é dividida em múltiplos fluxos de trabalho paralelos, cada um processando a informação de forma independente.
*   **Encadeamento Iterativo:** Repete um prompt ou uma cadeia até que uma condição específica seja atendida, sendo útil para refinar saídas (como na técnica de *Refinement*).
*   **Encadeamento Hierárquico:** Quebra uma tarefa grande em sub-tarefas menores, executadas hierarquicamente, onde as saídas de nível inferior alimentam as tarefas de nível superior [2].

A técnica é fundamental para a construção de fluxos de trabalho avançados em frameworks como o **LangChain**, que fornece ferramentas para gerenciar LLMs, definir prompts personalizados e conectá-los em cadeias reutilizáveis [1].

## Statistics

*   **Eficácia Comprovada:** Pesquisas recentes (como Sun et al., 2024) demonstram que o Prompt Chaining (orquestrando as fases de rascunho, crítica e refinamento através de prompts discretos) pode produzir um resultado mais favorável em tarefas como Sumarização de Texto em comparação com o *Stepwise Prompt* (que integra essas fases em um único prompt) [5].
*   **Adoção em Frameworks:** A técnica é um conceito fundamental em frameworks de orquestração de LLMs, como **LangChain** e **LlamaIndex**, indicando uma alta taxa de adoção na indústria e na comunidade de desenvolvimento de IA [1].
*   **Citações:** O artigo "Prompt Chaining or Stepwise Prompt? Refinement in Text Summarization" (2024) foi aceito no *Findings of ACL 2024*, um indicativo de sua relevância acadêmica [5].

## Features

*   **Modularidade:** Permite decompor tarefas complexas em etapas lógicas e independentes.
*   **Melhoria de Desempenho:** Aumenta a precisão e a qualidade da saída ao focar o LLM em subtarefas específicas.
*   **Gerenciamento de Contexto:** Ajuda a contornar as limitações da janela de contexto, processando informações em *chunks* ou etapas.
*   **Adaptabilidade:** Facilita a depuração e a otimização, pois cada etapa pode ser ajustada ou substituída individualmente.
*   **Orquestração de Fluxo de Trabalho:** Essencial para a criação de pipelines de IA sofisticados, como os utilizados em sistemas de Geração Aumentada por Recuperação (RAG) [1].

## Use Cases

*   **Processamento de Texto em Múltiplas Etapas:** Análise de *feedback* de clientes, onde a cadeia pode extrair palavras-chave, classificar o sentimento e gerar um resumo executivo [2].
*   **Geração Aumentada por Recuperação (RAG):** Uma cadeia pode ser usada para: 1) Recuperar documentos relevantes; 2) Gerar uma resposta inicial com base nos documentos; 3) Criticar e refinar a resposta para garantir a fidelidade ao texto de origem.
*   **Raciocínio Complexo e Resolução de Problemas:** Quebrar problemas matemáticos ou lógicos em etapas sequenciais, onde o resultado de cada passo é verificado antes de prosseguir (semelhante ao *Chain-of-Thought*, mas com prompts discretos).
*   **Criação de Conteúdo Refinado:** Uma cadeia pode gerar um rascunho, um segundo prompt pode criticar o rascunho e um terceiro prompt pode refinar o texto com base na crítica (Encadeamento Iterativo) [5].

## Integration

**Exemplo de Prompt Chaining (Sequencial) para Análise de Texto:**

**Passo 1: Extração de Entidades**
*   **Prompt:** "Extraia todas as entidades nomeadas (pessoas, organizações, locais) do seguinte texto: [TEXTO_DE_ENTRADA]"
*   **Output:** Lista de entidades.

**Passo 2: Classificação de Sentimento**
*   **Prompt:** "Com base nas entidades: [OUTPUT_PASSO_1], classifique o sentimento geral do texto original ([TEXTO_DE_ENTRADA]) como Positivo, Negativo ou Neutro. Justifique brevemente."
*   **Output:** Sentimento e Justificativa.

**Passo 3: Geração de Resumo Executivo**
*   **Prompt:** "Crie um resumo executivo de 50 palavras do texto original ([TEXTO_DE_ENTRADA]), focando nas entidades [OUTPUT_PASSO_1] e no sentimento [OUTPUT_PASSO_2]."
*   **Output:** Resumo Final.

**Melhores Práticas:**
1.  **Definir Limites Claros:** Cada prompt na cadeia deve ter um objetivo único e bem definido.
2.  **Formato de Saída Estruturado:** Use formatos como JSON ou XML para garantir que a saída de um prompt seja facilmente consumível como entrada para o próximo.
3.  **Tratamento de Erros:** Implemente mecanismos para lidar com falhas ou saídas inesperadas em qualquer etapa da cadeia.
4.  **Utilizar Frameworks:** Frameworks como **LangChain** ou **LlamaIndex** simplificam a orquestração e o gerenciamento de cadeias complexas [1].

## URL

https://www.ibm.com/think/tutorials/prompt-chaining-langchain