# Self-Consistency Prompting

## Description

**Self-Consistency Prompting** é uma estratégia de decodificação avançada que aprimora o raciocínio de modelos de linguagem grandes (LLMs), especialmente quando usados com o Chain-of-Thought (CoT) Prompting. Em vez de usar a decodificação gulosa (greedy decoding) que seleciona apenas um caminho de raciocínio, o Self-Consistency (SC) amostra um conjunto diversificado de caminhos de raciocínio e, em seguida, seleciona a resposta final mais consistente por meio de um sistema de votação majoritária. A intuição por trás dessa técnica é que problemas de raciocínio complexos geralmente admitem múltiplas formas de pensar que levam a uma única resposta correta. Ao explorar e agregar essas diversas perspectivas, o SC aumenta significativamente a precisão e a confiabilidade das respostas do LLM em tarefas que exigem raciocínio aritmético e de senso comum.

## Statistics

- **Artigo Original:** "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022).
- **Citações:** Mais de 3.100 citações (em Nov 2025), indicando alta relevância na pesquisa.
- **Ganhos de Desempenho (em comparação com CoT padrão):**
    - **GSM8K** (Matemática): **+17.9%**
    - **SVAMP** (Matemática): **+11.0%**
    - **AQuA** (Raciocínio): **+12.2%**
    - **StrategyQA** (Senso Comum): **+6.4%**
    - **ARC-challenge** (Raciocínio): **+3.9%**
- **Mecanismo:** Substitui a decodificação gulosa (greedy decoding) pela amostragem de múltiplos caminhos de raciocínio e votação majoritária.

## Features

- **Estratégia de Decodificação Avançada:** Substitui a decodificação gulosa ingênua (naive greedy decoding) usada no CoT.
- **Amostragem de Caminhos de Raciocínio:** Gera um conjunto diversificado de caminhos de raciocínio (cadeias de pensamento) a partir do LLM.
- **Votação Majoritária:** Seleciona a resposta final que é mais consistente entre todos os caminhos de raciocínio amostrados.
- **Melhora de Precisão:** Aumenta a precisão em tarefas de raciocínio complexas.
- **Redução de Viés:** Ajuda a mitigar vieses ao considerar diversas perspectivas.

## Use Cases

- **Resolução de Problemas Aritméticos:** Tarefas de matemática de nível escolar (GSM8K, SVAMP).
- **Raciocínio de Senso Comum:** Perguntas que exigem conhecimento e lógica do mundo real (StrategyQA).
- **Resolução de Problemas Complexos:** Qualquer tarefa que se beneficie da exploração de múltiplos caminhos de raciocínio para chegar a uma única resposta correta.
- **Aprimoramento de Modelos:** Usado para aumentar a precisão de modelos de linguagem grandes em tarefas de raciocínio.

## Integration

A técnica de Self-Consistency é implementada em duas etapas principais:

1.  **Geração de Múltiplos Caminhos de Raciocínio:** O prompt é executado várias vezes (N vezes) com amostragem de temperatura mais alta (por exemplo, `temperature > 0`) para gerar N cadeias de pensamento e respostas finais.

    **Exemplo de Prompt (Few-Shot CoT):**
    ```
    Q: Havia 15 árvores no bosque. Os trabalhadores do bosque plantarão árvores hoje. Depois que terminarem, haverá 21 árvores. Quantas árvores os trabalhadores plantaram hoje?
    A: Começamos com 15 árvores. Depois teremos 21 árvores. A diferença deve ser o número de árvores que eles plantaram. Então, eles devem ter plantado 21 - 15 = 6 árvores. A resposta é 6.

    Q: Quando eu tinha 6 anos, minha irmã tinha metade da minha idade. Agora tenho 70 anos. Quantos anos tem minha irmã?
    A:
    ```

2.  **Agregação e Votação:** As N respostas finais são coletadas, e a resposta que aparece com mais frequência (a mais consistente) é escolhida como a resposta final.

    **Exemplo de Resultados Amostrados:**
    - Caminho 1: "Minha irmã tinha 3 anos. Agora tenho 70, então ela tem 70 - 3 = 67. Resposta: 67."
    - Caminho 2: "A diferença de idade é 3 anos. Agora tenho 70, então ela tem 70 - 3 = 67. Resposta: 67."
    - Caminho 3: "70/2 = 35. Resposta: 35."
    - **Resposta Final (Votação):** 67 (2 votos)

**Melhores Práticas:**
- Use com tarefas que exigem raciocínio complexo (matemática, senso comum).
- Aumente o número de amostras (N) para maior robustez, mas monitore o custo computacional.
- Combine com o Chain-of-Thought (CoT) para obter o máximo de benefício.

## URL

https://arxiv.org/abs/2203.11171