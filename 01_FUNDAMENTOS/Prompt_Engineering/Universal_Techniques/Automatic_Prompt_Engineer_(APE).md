# Automatic Prompt Engineer (APE)

## Description

**Automatic Prompt Engineer (APE)** é uma estrutura para geração e seleção automática de instruções (prompts) para Modelos de Linguagem Grande (LLMs). O problema de geração de instruções é enquadrado como uma síntese de linguagem natural e abordado como um problema de otimização de caixa preta, utilizando LLMs para gerar e pesquisar um conjunto de soluções candidatas. O processo se desenrola em três etapas principais:
1.  **Geração de Candidatos:** Um LLM (o modelo de inferência) gera um conjunto de instruções candidatas a partir de demonstrações de entrada/saída (pares de exemplos).
2.  **Avaliação:** As instruções candidatas são executadas usando um LLM alvo, e uma função de pontuação (score function) é usada para avaliar a qualidade de cada instrução com base na precisão da execução.
3.  **Seleção:** A instrução com a pontuação mais alta é selecionada como o prompt ideal para a tarefa.

O APE demonstrou a capacidade de encontrar prompts que superam ou se igualam aos prompts criados por humanos, como a descoberta de uma variante do prompt Chain-of-Thought (CoT) que melhora o desempenho em tarefas de raciocínio.

## Statistics

- **Desempenho Humano-Nível:** O APE alcançou desempenho de nível humano ou superior em 24/24 tarefas de Indução de Instruções (Instruction Induction) e 17/21 tarefas BIG-Bench.
- **Melhoria em Raciocínio:** O prompt gerado pelo APE para CoT melhorou a precisão no benchmark **MultiArith** de 78,7% para **82,0%** e no **GSM8K** de 40,7% para **43,0%**, superando o prompt CoT padrão.
- **Citações:** O artigo original "Large Language Models Are Human-Level Prompt Engineers" (Zhou et al., 2022) possui mais de 1.400 citações (em Nov 2025), indicando alta relevância na pesquisa.

## Features

- **Otimização de Caixa Preta:** Não requer acesso aos gradientes ou parâmetros internos do LLM alvo.
- **Geração de Instruções:** Utiliza um LLM para gerar um conjunto diversificado de instruções candidatas.
- **Avaliação Baseada em Execução:** Pontua os prompts com base no desempenho real do LLM alvo em um conjunto de dados de validação.
- **Melhoria do CoT:** Capaz de descobrir prompts de Chain-of-Thought mais eficazes do que os prompts projetados por humanos.
- **Busca Iterativa:** Utiliza um processo de busca iterativa (Monte Carlo Search) para refinar e gerar novas instruções candidatas com base nas de melhor desempenho.

## Use Cases

- **Otimização de Prompts CoT:** Geração automática de prompts mais eficazes para o Chain-of-Thought (CoT) em tarefas de raciocínio.
- **Indução de Instruções:** Descoberta de instruções de linguagem natural para tarefas de processamento de linguagem natural (NLP) com base em exemplos de entrada/saída.
- **Melhoria de Desempenho:** Aumento da precisão e do desempenho de LLMs em uma ampla gama de tarefas, incluindo aritmética, senso comum e raciocínio lógico.
- **Pesquisa e Desenvolvimento:** Ferramenta para pesquisadores explorarem o espaço de prompts e descobrirem novas estratégias de prompt.

## Integration

**Exemplo de Prompt Otimizado (CoT):**
O APE descobriu um prompt de Chain-of-Thought (CoT) que superou o prompt "Let's think step by step" (Vamos pensar passo a passo) em tarefas de raciocínio:
> "Let's work this out in a step by step way to be sure we have the right answer." (Vamos resolver isso passo a passo para ter certeza de que temos a resposta certa.)

**Melhores Práticas:**
- **Demonstrações de Qualidade:** O desempenho do APE depende da qualidade dos pares de entrada/saída (demonstrações) fornecidos para a geração inicial dos prompts.
- **Função de Pontuação:** A escolha da função de pontuação é crucial, devendo medir com precisão o alinhamento entre o prompt, o conjunto de dados e o modelo.
- **Iteração:** Utilizar a busca iterativa (Monte Carlo Search) para explorar variações semânticas das instruções de melhor desempenho, garantindo que o processo não se restrinja a um conjunto inicial limitado.
- **Aplicação:** Ideal para tarefas onde a engenharia de prompt manual é complexa ou demorada, como tarefas de raciocínio complexo ou indução de instruções.

## URL

https://arxiv.org/pdf/2211.01910