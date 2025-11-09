# SQuAD (Stanford Question Answering Dataset)

## Description
O **Stanford Question Answering Dataset (SQuAD)** é um dos benchmarks mais influentes para sistemas de **Question Answering (QA)** e compreensão de leitura em Processamento de Linguagem Natural (PLN). O dataset é composto por perguntas elaboradas por colaboradores humanos sobre um conjunto de artigos da Wikipédia.

**Versões Principais:**

1.  **SQuAD 1.1:** Contém mais de **100.000 pares de perguntas e respostas** em mais de 500 artigos. A característica definidora desta versão é que a resposta para cada pergunta é sempre um **segmento de texto (span)** extraído diretamente do parágrafo de leitura correspondente (QA Extrativo).
2.  **SQuAD 2.0:** É uma versão mais desafiadora que combina as perguntas do SQuAD 1.1 com mais de **50.000 perguntas não respondíveis** criadas de forma adversária por colaboradores. Para ter um bom desempenho no SQuAD 2.0, os modelos devem não apenas responder às perguntas quando possível, mas também determinar quando o parágrafo não contém a resposta e se abster de responder.

O SQuAD é amplamente utilizado para treinar e avaliar a capacidade de modelos de PLN de ler um texto e extrair a resposta correta para uma pergunta.

## Statistics
**Versões:** SQuAD 1.1 e SQuAD 2.0.
**SQuAD 1.1:**
*   **Amostras:** Mais de 100.000 pares de perguntas-respostas.
*   **Artigos:** Mais de 500 artigos da Wikipédia.
**SQuAD 2.0:**
*   **Amostras:** Mais de 150.000 perguntas no total.
*   **Perguntas Não Respondíveis:** Mais de 50.000.
*   **Tamanho (v2.0):** Treino (40 MB), Desenvolvimento (4 MB).

## Features
*   **Formato QA Extrativo:** As respostas são segmentos de texto (spans) do contexto.
*   **Contexto Base:** Artigos da Wikipédia.
*   **SQuAD 2.0:** Inclui mais de 50.000 perguntas não respondíveis, exigindo que os modelos determinem a ausência de resposta.
*   **Métricas de Avaliação:** Principalmente **Exact Match (EM)** e **F1 Score**.
*   **Licença:** Distribuído sob a licença **CC BY-SA 4.0**.

## Use Cases
*   **Treinamento de Sistemas de QA Extrativo:** Principal uso para desenvolver modelos que localizam a resposta exata em um texto.
*   **Avaliação de Compreensão de Leitura:** Serve como um benchmark padrão para medir a capacidade de máquinas de entender e processar texto.
*   **Desenvolvimento de Modelos Robustos (SQuAD 2.0):** Essencial para treinar modelos que conseguem diferenciar perguntas respondíveis de não respondíveis, crucial para aplicações em ambientes reais.
*   **Pesquisa em PLN:** Utilizado para testar novas arquiteturas de modelos de linguagem (como BERT, T5, etc.) e técnicas de transfer learning.
*   **Assistentes Virtuais e Chatbots:** A tecnologia desenvolvida com o SQuAD é a base para sistemas que respondem a perguntas baseadas em documentos de conhecimento.

## Integration
O SQuAD é facilmente acessível e pode ser integrado em projetos de PLN de várias maneiras:

1.  **Hugging Face Datasets:** A maneira mais recomendada e moderna de acessar o dataset, permitindo o carregamento direto com poucas linhas de código Python:
    ```python
    from datasets import load_dataset
    # Para SQuAD 1.1
    squad_v1 = load_dataset("squad")
    # Para SQuAD 2.0
    squad_v2 = load_dataset("squad_v2")
    ```
2.  **Download Direto (JSON):** Os arquivos JSON originais podem ser baixados do site oficial para uso manual ou em frameworks que não suportam o carregamento automático.
    *   **SQuAD 2.0 Treino:** `train-v2.0.json` (40 MB)
    *   **SQuAD 2.0 Desenvolvimento:** `dev-v2.0.json` (4 MB)
3.  **Kaggle:** O dataset também está disponível no Kaggle, facilitando o uso em seus notebooks.

**Instruções de Uso:** O dataset é tipicamente usado para ajustar (fine-tuning) modelos de linguagem pré-treinados (como BERT, RoBERTa, ELECTRA) para a tarefa de Question Answering. O processo envolve alimentar o modelo com o parágrafo de contexto e a pergunta, e o modelo deve prever os índices de início e fim do span da resposta no texto. Para o SQuAD 2.0, o modelo também deve prever se a pergunta é "não respondível".

## URL
[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
