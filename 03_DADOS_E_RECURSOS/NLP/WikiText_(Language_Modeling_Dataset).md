# WikiText (Language Modeling Dataset)

## Description
O **WikiText** é um conjunto de dados de modelagem de linguagem amplamente utilizado, extraído de artigos verificados ("Good" e "Featured") da Wikipédia. Ele se destaca por reter a capitalização, pontuação e números originais, ao contrário de conjuntos de dados mais antigos como o Penn Treebank (PTB), tornando-o mais adequado para modelos que exploram dependências de longo prazo. O dataset é oferecido em duas versões principais: **WikiText-2** (menor) e **WikiText-103** (maior). Cada versão possui variantes "raw" (bruta, para trabalho em nível de caractere) e "v1" (para trabalho em nível de palavra, onde tokens fora do vocabulário são substituídos por `<unk>`). A versão WikiText-103 contém mais de 100 milhões de tokens.

**English Description:** The **WikiText** is a widely used language modeling dataset, extracted from verified ("Good" and "Featured") articles on Wikipedia. It stands out for retaining original capitalization, punctuation, and numbers, unlike older datasets like the Penn Treebank (PTB), making it more suitable for models that explore long-term dependencies. The dataset is offered in two main versions: **WikiText-2** (smaller) and **WikiText-103** (larger). Each version has "raw" variants (for character-level work) and "v1" variants (for word-level work, where out-of-vocabulary tokens are replaced with `<unk>`). The WikiText-103 version contains over 100 million tokens.

## Statistics
O dataset é dividido em duas versões principais, cada uma com variantes "raw" e "v1".

| Versão | Tokens (Aprox.) | Vocabulário (Aprox.) | Tamanho do Download (Hugging Face) | Linhas (Split de Treino) |
| :--- | :--- | :--- | :--- | :--- |
| **WikiText-103** | > 100 Milhões | 260 Mil | 190 MB (v1) / 192 MB (raw) | 1.801.350 |
| **WikiText-2** | 2 Milhões | 33 Mil | 4.5 MB (v1) / 4.7 MB (raw) | 36.718 |

**Estatísticas Detalhadas (WikiText-103-v1):**
*   **Split de Treino:** 1.801.350 linhas
*   **Split de Validação:** 3.760 linhas
*   **Split de Teste:** 4.358 linhas
*   **Tamanho Total (Gerado):** 738.27 MB

**English Statistics:** The dataset is split into two main versions, each with "raw" and "v1" variants.

| Version | Tokens (Approx.) | Vocabulary (Approx.) | Download Size (Hugging Face) | Rows (Train Split) |
| :--- | :--- | :--- | :--- | :--- |
| **WikiText-103** | > 100 Million | 260 Thousand | 190 MB (v1) / 192 MB (raw) | 1,801,350 |
| **WikiText-2** | 2 Million | 33 Thousand | 4.5 MB (v1) / 4.7 MB (raw) | 36,718 |

**Detailed Statistics (WikiText-103-v1):**
*   **Train Split:** 1,801,350 rows
*   **Validation Split:** 3,760 rows
*   **Test Split:** 4,358 rows
*   **Total Size (Generated):** 738.27 MB

## Features
*   **Fidelidade ao Texto Original:** Mantém a capitalização, pontuação e números originais, oferecendo um corpus mais realista.
*   **Dependências de Longo Prazo:** Composto por artigos completos, é ideal para treinar modelos que se beneficiam de contexto de longo alcance.
*   **Variantes de Tamanho:** Oferece duas escalas, WikiText-2 e WikiText-103, para diferentes necessidades de treinamento.
*   **Variantes de Tokenização:** Inclui versões "raw" (brutas) para modelagem em nível de caractere e versões processadas para modelagem em nível de palavra.
*   **Vocabulário Extenso:** Possui um vocabulário significativamente maior do que conjuntos de dados mais antigos.

**English Features:**
*   **Fidelity to Original Text:** Retains original capitalization, punctuation, and numbers, offering a more realistic corpus.
*   **Long-Term Dependencies:** Composed of full articles, it is ideal for training models that benefit from long-range context.
*   **Size Variants:** Offers two scales, WikiText-2 and WikiText-103, for different training needs.
*   **Tokenization Variants:** Includes "raw" versions for character-level modeling and processed versions for word-level modeling.
*   **Extensive Vocabulary:** Has a significantly larger vocabulary than older datasets.

## Use Cases
*   **Modelagem de Linguagem (Language Modeling):** Principal caso de uso, servindo como benchmark para modelos de linguagem baseados em palavras e caracteres.
*   **Avaliação de Dependências de Longo Prazo:** Ideal para testar a capacidade de modelos de manter o contexto em sequências longas, devido à sua composição de artigos completos.
*   **Pré-treinamento de Modelos de Linguagem:** Utilizado para pré-treinar arquiteturas como LSTMs, Transformers e QRNNs.
*   **Pesquisa em NLP:** Usado para comparar o desempenho de novos modelos e técnicas de processamento de linguagem natural.

**English Use Cases:**
*   **Language Modeling:** Primary use case, serving as a benchmark for word-based and character-based language models.
*   **Long-Term Dependency Evaluation:** Ideal for testing the ability of models to maintain context over long sequences, due to its composition of full articles.
*   **Language Model Pre-training:** Used to pre-train architectures such as LSTMs, Transformers, and QRNNs.
*   **NLP Research:** Used to compare the performance of new models and natural language processing techniques.

## Integration
O dataset é facilmente acessível e carregável através da biblioteca **Hugging Face Datasets** em Python.

**Instruções de Uso (Python/Hugging Face):**

```python
from datasets import load_dataset

# Para carregar a versão WikiText-103 (processada, nível de palavra)
dataset_103 = load_dataset("wikitext", "wikitext-103-v1")

# Para carregar a versão WikiText-2 (processada, nível de palavra)
dataset_2 = load_dataset("wikitext", "wikitext-2-v1")

# Para carregar a versão bruta (raw) do WikiText-103
dataset_103_raw = load_dataset("wikitext", "wikitext-103-raw-v1")

# Acessar as divisões (splits)
train_data = dataset_103["train"]
validation_data = dataset_103["validation"]
test_data = dataset_103["test"]
```

**English Integration:** The dataset is easily accessible and loadable through the **Hugging Face Datasets** library in Python.

**Usage Instructions (Python/Hugging Face):**

```python
from datasets import load_dataset

# To load the WikiText-103 version (processed, word-level)
dataset_103 = load_dataset("wikitext", "wikitext-103-v1")

# To load the WikiText-2 version (processed, word-level)
dataset_2 = load_dataset("wikitext", "wikitext-2-v1")

# To load the raw version of WikiText-103
dataset_103_raw = load_dataset("wikitext", "wikitext-103-raw-v1")

# Access the splits
train_data = dataset_103["train"]
validation_data = dataset_103["validation"]
test_data = dataset_103["test"]
```

## URL
[https://huggingface.co/datasets/Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
