# Prompt Compression (LLMLingua)

## Description
**LLMLingua** é uma técnica de compressão de prompt de "grosso a fino" (coarse-to-fine) desenvolvida pela Microsoft Research para acelerar a inferência de Large Language Models (LLMs) e reduzir custos, especialmente em cenários com prompts longos, como Chain-of-Thought (CoT) e In-Context Learning (ICL) [1].

A técnica se baseia na premissa de que a linguagem natural é inerentemente redundante e que os LLMs podem inferir a partir de prompts comprimidos, mesmo que sejam difíceis de entender para humanos. O método utiliza um Small Language Model (SLM) auxiliar (como GPT2-small ou LLaMA-7B) para calcular a perplexidade (PPL) e identificar *tokens* e sentenças redundantes no prompt.

O processo de compressão envolve três componentes principais [2]:
1.  **Controlador de Orçamento (Budget Controller):** Define a taxa de compressão alvo e aloca o "orçamento" de *tokens* para diferentes partes do prompt, preservando a integridade semântica.
2.  **Compressão Iterativa Nível-Token:** Remove *tokens* redundantes de forma iterativa, modelando a interdependência entre o conteúdo comprimido.
3.  **Alinhamento de Distribuição:** Utiliza *instruction tuning* para alinhar o SLM auxiliar com a distribuição de linguagem do LLM alvo (caixa-preta), garantindo que a compressão seja eficaz para o modelo final.

A LLMLingua demonstrou ser capaz de alcançar até **20x de compressão** com pouca perda de desempenho em tarefas como raciocínio e ICL, resultando em uma aceleração de 1.7x a 5.7x na latência de ponta a ponta [3].

## Examples
```
**Exemplo 1: Compressão de Contexto RAG (Recuperação Aumentada)**
*   **Prompt Original (Entrada para LLMLingua):** "Contexto: [10.000 tokens de documentos recuperados sobre a fusão da Empresa X e Y]. Pergunta: Qual foi o principal motivo estratégico para a fusão, de acordo com o documento?"
*   **Prompt Comprimido (Saída para LLM):** "Contexto: [500 tokens comprimidos]. Pergunta: Principal motivo estratégico para a fusão?"

**Exemplo 2: Compressão de Exemplos ICL (In-Context Learning)**
*   **Prompt Original (Entrada para LLMLingua):** "Exemplo 1: [Problema de matemática complexo e solução detalhada com CoT]. Exemplo 2: [Outro problema e solução]. Pergunta: Resolva o seguinte problema: [Novo problema]."
*   **Prompt Comprimido (Saída para LLM):** "Exemplo 1: [CoT comprimido]. Exemplo 2: [CoT comprimido]. Pergunta: Resolva: [Novo problema]."

**Exemplo 3: Compressão de Diálogo Longo (Chatbot)**
*   **Prompt Original (Entrada para LLMLingua):** "Histórico de Conversa: [50 turnos de conversa]. Usuário: Quero saber a política de devolução para itens eletrônicos comprados há mais de 30 dias."
*   **Prompt Comprimido (Saída para LLM):** "Histórico: [Essência comprimida dos 50 turnos]. Usuário: Política de devolução eletrônicos > 30 dias."

**Exemplo 4: Compressão de Resumo de Reunião**
*   **Prompt Original (Entrada para LLMLingua):** "Transcrição da Reunião: [20.000 tokens de transcrição]. Instrução: Gere um resumo executivo com as 3 principais decisões e os responsáveis."
*   **Prompt Comprimido (Saída para LLM):** "Transcrição: [1.000 tokens de transcrição comprimida]. Instrução: Resumo executivo: 3 decisões, responsáveis."

**Exemplo 5: Compressão de Instruções Detalhadas**
*   **Prompt Original (Entrada para LLMLingua):** "Instruções: Você é um especialista em Python. Responda de forma concisa, use apenas a biblioteca Pandas e o formato de saída deve ser JSON. A tarefa é: [Descrição da tarefa]."
*   **Prompt Comprimido (Saída para LLM):** "Instruções: Especialista Python. Resposta concisa. Use Pandas. Saída JSON. Tarefa: [Descrição da tarefa]."
```

## Best Practices
**1. Priorize a Informação Chave:** Utilize a LLMLingua para comprimir a parte do prompt que contém informações contextuais (como exemplos de ICL ou documentos RAG) e mantenha a instrução principal e a pergunta intactas.
**2. Use um Modelo Pequeno Otimizado:** A eficácia da LLMLingua depende de um modelo pequeno (como GPT2-small ou LLaMA-7B) para calcular a perplexidade e identificar a redundância. Certifique-se de que o modelo auxiliar esteja alinhado com o LLM alvo.
**3. Monitore a Taxa de Compressão:** Comece com taxas de compressão mais baixas (por exemplo, 5x) e aumente gradualmente, monitorando a métrica de desempenho (como EM ou precisão) para encontrar o ponto ideal de equilíbrio entre custo/velocidade e qualidade.
**4. Aproveite a Recuperabilidade:** Em cenários críticos, use um LLM poderoso (como GPT-4) para descompactar o prompt comprimido, garantindo que nenhuma informação essencial tenha sido perdida.
**5. Integração RAG:** Integre a LLMLingua em *frameworks* RAG (como LlamaIndex ou LangChain) para comprimir os documentos recuperados antes de passá-los para o LLM, otimizando o custo e a latência.

## Use Cases
**1. Otimização de Custo e Latência em APIs:** Redução drástica do número de *tokens* de entrada enviados para LLMs de caixa-preta (como GPT-4 ou Claude) via API, resultando em economia de custos e menor latência de resposta.
**2. Frameworks RAG (Retrieval-Augmented Generation):** Compressão de documentos e trechos recuperados antes de serem inseridos no prompt do LLM, permitindo a inclusão de mais contexto relevante e melhorando a densidade de informação.
**3. Aceleração de In-Context Learning (ICL):** Compressão de exemplos de ICL longos e detalhados (incluindo cadeias de pensamento - CoT) para manter a capacidade de raciocínio do LLM com um prompt menor.
**4. Compressão de KV Cache:** Utilização da técnica para comprimir o *Key-Value Cache* durante a inferência, o que melhora a velocidade de decodificação e permite contextos mais longos.
**5. Resumo de Contextos Longos:** Aplicação em cenários de contexto muito longo, como transcrições de reuniões, históricos de conversas ou documentos extensos, para extrair a essência e facilitar o resumo ou a resposta a perguntas.

## Pitfalls
**1. Perda de Informação Crítica:** A compressão excessiva (taxas muito altas, como 20x sem validação) pode remover *tokens* ou sentenças que, embora pareçam redundantes, são cruciais para a precisão da resposta do LLM, especialmente em tarefas de raciocínio complexo (CoT).
**2. Desalinhamento do SLM:** Se o Small Language Model (SLM) auxiliar não estiver bem alinhado com o LLM alvo (caixa-preta), a compressão pode ser ineficaz ou prejudicial, pois o SLM pode não identificar corretamente a redundância do ponto de vista do LLM.
**3. Latência da Compressão:** Embora a LLMLingua reduza a latência de inferência do LLM, o processo de compressão em si introduz uma latência adicional. Em prompts muito curtos, o tempo gasto na compressão pode anular o ganho de velocidade na inferência.
**4. Dificuldade de Depuração:** O prompt comprimido é quase ilegível para humanos. Isso torna a depuração e a otimização do prompt muito mais difíceis, pois o pesquisador não consegue inspecionar facilmente o que o LLM está realmente recebendo.
**5. Custo Adicional do SLM:** A execução do SLM auxiliar para realizar a compressão adiciona um custo computacional (e potencialmente financeiro, se for um serviço pago) que deve ser considerado no cálculo do custo-benefício total.

## URL
[https://arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736)
