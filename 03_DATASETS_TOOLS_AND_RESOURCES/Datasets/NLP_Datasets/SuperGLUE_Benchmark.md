# SuperGLUE Benchmark

## Description
O **SuperGLUE (Super General Language Understanding Evaluation)** é um benchmark de avaliação de modelos de Linguagem Natural (NLP) projetado para ser mais desafiador que seu antecessor, o GLUE. Ele foi criado para medir o progresso em sistemas de compreensão de linguagem de propósito geral, focando em tarefas que exigem raciocínio mais profundo, inferência e compreensão contextual. O benchmark é composto por um conjunto de tarefas de compreensão de linguagem mais difíceis, recursos aprimorados e um leaderboard público. O objetivo é fornecer uma métrica única que resuma o progresso em um conjunto diversificado de tarefas de NLP, especialmente após os modelos terem atingido o desempenho humano no benchmark GLUE original.

## Statistics
O SuperGLUE é um benchmark composto por 10 tarefas (8 principais e 2 de diagnóstico), cada uma com seu próprio conjunto de dados.
*   **Tamanho Total do Download (TFDS):** Aproximadamente 733.32 KiB.
*   **Tamanho Total do Dataset (TFDS):** Aproximadamente 2.15 MiB.
*   **Versão:** A versão final do benchmark foi lançada em 2019, mas continua sendo um padrão de avaliação relevante, com modelos sendo submetidos e avaliados continuamente.
*   **Amostras:** O número de amostras varia por tarefa. Por exemplo, o BoolQ possui mais de 15.000 exemplos de treinamento e desenvolvimento. O conjunto completo de tarefas totaliza dezenas de milhares de exemplos.

## Features
O SuperGLUE é composto por 8 tarefas principais de compreensão de linguagem e 2 tarefas de diagnóstico para análise de erros. As tarefas principais incluem:
*   **BoolQ:** Resposta a perguntas de sim/não baseadas em um parágrafo.
*   **CB (CommitmentBank):** Determinação da relação de inferência textual (entailment, contradição, neutro) entre duas sentenças.
*   **COPA (Choice of Plausible Alternatives):** Escolha da alternativa mais plausível para um dado cenário (causa ou efeito).
*   **MultiRC (Multi-Sentence Reading Comprehension):** Resposta a perguntas sobre um texto, onde a resposta pode ser uma ou mais sentenças.
*   **ReCoRD (Reading Comprehension with Commonsense Reasoning):** Preenchimento de lacunas em um texto com base em raciocínio de senso comum.
*   **RTE (Recognizing Textual Entailment):** Determinação se uma sentença implica logicamente outra.
*   **WiC (Words in Context):** Determinação se uma palavra aparece com o mesmo sentido em duas sentenças diferentes.
*   **WSC (Winograd Schema Challenge):** Resolução de ambiguidades de referência pronominal que exigem raciocínio de senso comum.

As tarefas de diagnóstico são AX-b (Broadcoverage Diagnostics) e AX-g (WinoGender Schema Diagnostics). O benchmark é caracterizado por exigir modelos que demonstrem capacidades de inferência e raciocínio mais robustas.

## Use Cases
*   **Avaliação de Modelos de Linguagem:** É o principal caso de uso, servindo como um teste rigoroso para modelos de linguagem de propósito geral (LLMs) e modelos pré-treinados (como BERT, RoBERTa, T5, etc.).
*   **Pesquisa em NLP:** Utilizado por pesquisadores para desenvolver e testar novas arquiteturas e técnicas de transferência de aprendizado em tarefas de compreensão de linguagem mais complexas.
*   **Análise de Erros:** As tarefas de diagnóstico (AX-b e AX-g) são usadas para realizar análises qualitativas e de erros, ajudando a entender as deficiências dos modelos.
*   **Comparação de Desempenho:** Serve como um leaderboard público para comparar o desempenho de diferentes sistemas de NLP em um conjunto padronizado de tarefas.

## Integration
O dataset SuperGLUE pode ser acessado e utilizado de várias maneiras:
1.  **Página Oficial:** O dataset completo pode ser baixado diretamente da página de tarefas do SuperGLUE (URL: `https://super.gluebenchmark.com/tasks`).
2.  **Script de Download:** O site oficial fornece um script de download (parte do toolkit `jiant`) para obter os dados.
3.  **Hugging Face Datasets:** Para uso em projetos de NLP modernos, o dataset está disponível no Hugging Face Hub (ex: `Hyukkyu/superglue` ou `aps/super_glue`), permitindo o carregamento fácil via código Python:
    ```python
    from datasets import load_dataset
    # Para carregar uma tarefa específica, como BoolQ
    dataset = load_dataset("super_glue", "boolq")
    ```
4.  **TensorFlow Datasets:** O dataset também está disponível no catálogo do TensorFlow Datasets.

A integração é facilitada por ferramentas e bibliotecas de NLP amplamente utilizadas. Recomenda-se o uso das versões hospedadas no Hugging Face ou TensorFlow para maior conveniência.

## URL
[https://super.gluebenchmark.com/](https://super.gluebenchmark.com/)
