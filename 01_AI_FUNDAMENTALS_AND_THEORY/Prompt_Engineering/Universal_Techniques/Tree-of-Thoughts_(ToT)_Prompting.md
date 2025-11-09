# Tree-of-Thoughts (ToT) Prompting

## Description

O **Tree-of-Thoughts (ToT) Prompting** é uma técnica avançada de engenharia de prompt que generaliza o método Chain-of-Thought (CoT), permitindo que Large Language Models (LLMs) realizem um processo de tomada de decisão mais deliberado e estratégico. Em vez de seguir um único caminho linear de raciocínio (como no CoT), o ToT estrutura o processo de resolução de problemas como uma **árvore de busca**, onde cada nó representa um "pensamento" (uma unidade coerente de texto, como um passo intermediário). Isso permite que o LLM explore múltiplas vias de raciocínio em paralelo, avalie a promessa de cada caminho e faça escolhas globais, incluindo a capacidade de olhar à frente (*lookahead*) ou retroceder (*backtracking*) quando necessário. O ToT é particularmente eficaz em tarefas que exigem planejamento não trivial, busca e raciocínio complexo, onde as decisões iniciais são cruciais para o sucesso final [1] [2].

## Statistics

- **Aumento de Desempenho:** O ToT demonstrou um aumento significativo na capacidade de resolução de problemas em comparação com o Chain-of-Thought (CoT).
- **Game of 24:** No desafio "Game of 24", o GPT-4 com CoT resolveu apenas **4%** das tarefas, enquanto o GPT-4 com ToT alcançou uma taxa de sucesso de **74%** [1].
- **Outros Ganhos:** O ToT também mostrou melhorias substanciais em tarefas como **Escrita Criativa** e **Mini Crosswords**, que exigem planejamento e busca não triviais [1].
- **Citação:** O artigo original "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023) é uma referência fundamental na área, com alta taxa de citação desde sua publicação [1].

## Features

- **Exploração de Múltiplos Caminhos:** Permite que o LLM gere e avalie diversas sequências de raciocínio (pensamentos) em paralelo, em vez de se limitar a uma única cadeia linear.
- **Busca Estratégica:** Utiliza algoritmos de busca (como **Depth-First Search (DFS)**, **Breadth-First Search (BFS)** e **Beam Search**) para navegar na árvore de pensamentos, selecionando os caminhos mais promissores.
- **Autoavaliação (Self-Evaluation):** O LLM avalia a qualidade e a progressão de cada "pensamento" em direção à solução final, usando raciocínio baseado em linguagem para guiar a busca.
- **Decisão Deliberada:** Facilita a tomada de decisões estratégicas, permitindo que o modelo olhe à frente e retroceda, superando as limitações de decisão token-a-token do CoT [1] [2].

## Use Cases

- **Resolução de Quebra-Cabeças e Jogos:** Especialmente eficaz em jogos que exigem planejamento e busca, como o **Game of 24** e **Sudoku** (em variações do framework) [1] [2].
- **Escrita Criativa:** Geração de narrativas ou textos que exigem coerência e planejamento de longo prazo.
- **Raciocínio Multi-Etapas:** Tarefas que envolvem múltiplas variáveis inter-relacionadas e onde a decisão em uma etapa afeta criticamente as etapas subsequentes.
- **Planejamento Estratégico:** Simulação de processos de tomada de decisão que requerem olhar à frente e avaliar diferentes cenários (ex: planejamento de negócios, análise de mercado) [2].

## Integration

A implementação do ToT pode ser feita de duas formas principais: via código (integrando o LLM com algoritmos de busca) ou via prompt (instruindo o LLM a simular o processo de busca).

**Exemplo de Prompt Simples (Simulação ToT):**
Uma abordagem simplificada, proposta por Dave Hubert, instrui o LLM a simular um processo de deliberação em grupo para resolver problemas complexos [2]:
```
Imagine que três especialistas diferentes estão respondendo a esta pergunta.
Todos os especialistas escreverão 1 passo de seu raciocínio,
e então compartilharão com o grupo.
Em seguida, todos os especialistas prosseguirão para o próximo passo, e assim por diante.
Se algum especialista perceber que está errado em algum momento, ele deve sair.
A pergunta é... [Insira a pergunta complexa aqui]
```

**Melhores Práticas:**
- **Usar para Problemas Complexos:** Aplique ToT em tarefas que exigem planejamento, estratégia e onde o CoT falha (ex: quebra-cabeças, raciocínio multi-etapas).
- **Estruturação do Prompt:** Para implementações via código, o prompt deve ser estruturado para gerar **pensamentos coerentes** (não apenas tokens), **avaliar o estado** (heurística) e **selecionar o próximo passo** [1].
- **Algoritmos de Busca:** A escolha do algoritmo (DFS, BFS, Beam Search) deve ser adaptada à natureza do problema. DFS é útil para explorar profundamente um caminho, enquanto BFS/Beam Search são melhores para manter a diversidade de opções [2].

## URL

https://arxiv.org/abs/2305.10601