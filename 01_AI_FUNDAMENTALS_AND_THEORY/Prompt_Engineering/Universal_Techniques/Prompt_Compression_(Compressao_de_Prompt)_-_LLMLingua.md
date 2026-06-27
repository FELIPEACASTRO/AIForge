# Prompt Compression (Compressão de Prompt) - LLMLingua

## Description

A **Compressão de Prompt** é uma técnica de engenharia de prompt que visa reduzir o comprimento (número de tokens) do prompt de entrada para Large Language Models (LLMs) enquanto preserva a informação essencial e a intenção da tarefa. O objetivo principal é mitigar os desafios de prompts longos, como o aumento do custo de inferência, a redução da velocidade de processamento e o consumo elevado de memória. A técnica é crucial para otimizar aplicações como a Geração Aumentada por Recuperação (RAG), onde o contexto recuperado pode ser excessivamente longo.

A compressão pode ser categorizada em dois métodos principais:
1.  **Métodos de Prompt Rígido (Hard Prompt Methods):** Envolvem a remoção direta de tokens de baixa informação (Filtragem) ou a reescrita do prompt para concisão (Paráfrase). Exemplos incluem SelectiveSentence e LLMLingua (que usa um LLM menor para identificar e remover tokens não essenciais).
2.  **Métodos de Prompt Suave (Soft Prompt Methods):** Comprimem o texto em um número menor de tokens especiais ou *embeddings* (como GIST ou ICAE), que são então usados pelo LLM principal.

A implementação mais notável e bem-sucedida é o **LLMLingua**, que utiliza um LLM menor e mais rápido (como GPT2-small ou LLaMA-7B) para calcular a perplexidade e a importância de cada token, removendo aqueles que menos contribuem para a informação essencial.

## Statistics

- **Taxa de Compressão:** O LLMLingua alcança até **20x de taxa de compressão** (redução de 95% dos tokens) com perda mínima de desempenho em tarefas de ICL (In-Context Learning) [1] [2].
- **Redução de Custo:** Redução de custo de inferência em até **3.5x** [3] ou **80%** em aplicações RAG (Retrieval-Augmented Generation) [4].
- **Melhoria de Desempenho:** O LongLLMLingua, uma variação da técnica, demonstrou uma melhoria de desempenho de **17.1%** com uma compressão de **4x** em contextos longos [1].
- **Retenção de Capacidade:** Em estudos empíricos, o LLM reteve entre **62.26% e 72.89%** de seu desempenho original mesmo com compressão significativa [5].
- **Citação:** A técnica LLMLingua (Compressing Prompts for Accelerated Inference of Large Language Models) foi publicada em 2023 [6].

**Referências:**
[1] https://www.llmlingua.com/
[2] https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03
[3] https://builder.aws.com/content/2n9wWygDkfoZd74eAsaBNtEjZON/prompt-compression-using-amazon-bedrock-reduce-rag-costs
[4] https://towardsdatascience.com/how-to-cut-rag-costs-by-80-using-prompt-compression-877a07c6bedb/
[5] https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03
[6] https://arxiv.org/html/2310.05736v2

## Features

- **Redução de Custo e Latência:** Diminui drasticamente o número de tokens de entrada, resultando em custos mais baixos por chamada de API e inferência mais rápida.
- **Preservação de Desempenho:** Mantém a capacidade de resposta do LLM principal, garantindo que a informação crítica para a tarefa seja retida.
- **Otimização de Contexto:** Permite que mais contexto relevante caiba na janela de contexto do LLM, melhorando o desempenho em tarefas de Geração Aumentada por Recuperação (RAG) e In-Context Learning (ICL).
- **Abordagem Híbrida (LLMLingua):** Utiliza um modelo de compressão menor (como GPT2-small) para pré-processar o prompt, tornando a técnica agnóstica ao modelo principal (pode ser usada com GPT-4, Claude, etc.).
- **Compressão de Múltiplos Componentes:** O LLMLingua permite a compressão de diferentes partes do prompt, como `Instruções`, `Contexto` e `Exemplos`.

## Use Cases

- **Geração Aumentada por Recuperação (RAG):** Otimização do contexto recuperado de bases de dados vetoriais, onde a informação pode ser redundante ou excessiva. A compressão garante que apenas os trechos mais relevantes sejam passados ao LLM, reduzindo custos e latência.
- **In-Context Learning (ICL):** Compressão de exemplos de ICL (few-shot examples) dentro do prompt, permitindo que mais exemplos caibam na janela de contexto e melhorando a capacidade de aprendizado do modelo.
- **Aplicações de Diálogo e Chatbots:** Redução do histórico de conversas para manter o contexto relevante sem exceder o limite de tokens ou aumentar os custos de forma proibitiva.
- **Inferência de Baixa Latência:** Uso em aplicações que exigem respostas rápidas, como assistentes virtuais em tempo real, onde a redução do tempo de processamento do prompt é crítica.
- **Otimização de Custos:** Implementação em qualquer aplicação de LLM onde o custo por token de entrada é uma preocupação significativa.

## Integration

**Melhores Práticas e Exemplos de Integração (LLMLingua):**

1.  **Divisão do Prompt:** Para obter a melhor compressão, o prompt deve ser dividido em componentes: `Instruções`, `Contexto` e `Perguntas`. O LLMLingua permite que o usuário defina a importância relativa de cada seção.
2.  **Uso em RAG:** A compressão é ideal para o contexto recuperado em aplicações RAG. O contexto longo e redundante é comprimido antes de ser passado ao LLM principal, reduzindo o custo em até 80%.
3.  **Exemplo de Código (Conceitual):**
    ```python
    from llmlingua import LLMLingua

    # Inicializa o compressor com um modelo pequeno
    llm_lingua = LLMLingua(model_name="gpt2")

    long_prompt = {
        "instruction": "Responda à pergunta com base no contexto fornecido.",
        "context": "O contexto recuperado de um banco de dados vetorial, contendo muitos parágrafos sobre a história da IA e o desenvolvimento de LLMs...",
        "question": "Qual é o principal benefício da compressão de prompt?"
    }

    # Comprime o prompt
    compressed_prompt = llm_lingua.compress_prompt(
        long_prompt,
        rate=0.5, # Taxa de compressão desejada (50% do tamanho original)
        force_context_compress=True
    )

    # O prompt comprimido é então enviado ao LLM principal (e.g., GPT-4)
    # response = llm_main.generate(compressed_prompt)
    ```
4.  **Configuração de Taxa:** Comece com taxas de compressão mais baixas (e.g., 2x ou 4x) e aumente gradualmente, monitorando a queda de desempenho. O LLMLingua alcança o melhor equilíbrio entre compressão e desempenho em torno de 4x.

## URL

https://www.llmlingua.com/