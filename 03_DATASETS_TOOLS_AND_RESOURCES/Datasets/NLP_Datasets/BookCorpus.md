# BookCorpus

## Description
O **BookCorpus** é um corpus de texto massivo, originalmente composto por cerca de 11.038 livros auto-publicados, extraídos da plataforma de distribuição de e-books independentes Smashwords. Foi criado para o treinamento de modelos de linguagem influentes, como o **BERT** e seus derivados, sendo notável por fornecer texto de formato longo, o que é crucial para o aprendizado de dependências de longo alcance em modelos de Processamento de Linguagem Natural (PLN). Uma análise retrospectiva de 2021 destacou que o dataset possui "dívida de documentação", contém restrições de direitos autorais, inclui milhares de livros duplicados e apresenta um viés significativo na representação de gêneros (por exemplo, Romance e Fantasia são super-representados). A versão mais limpa e amplamente utilizada atualmente é o **BookCorpusOpen**, que tenta mitigar essas deficiências.

## Statistics
- **Número de Livros (Versão Analisada em 2021):** 11.038 (com apenas 7.185 livros únicos).
- **Número de Sentenças:** 74.004.228
- **Número de Palavras:** 984.346.357
- **Tamanho do Download (Versão Hugging Face):** 4.61 GB (arquivos de dataset) / 3.02 GB (arquivos Parquet convertidos).
- **Versões Notáveis:**
    - **Original BookCorpus (2015):** Versão inicial usada para treinar o BERT.
    - **BookCorpusOpen:** Uma versão mais limpa e popular que visa resolver problemas de duplicação e direitos autorais.
    - **Versões em Repositórios (e.g., Hugging Face):** Variantes re-hospedadas e pré-processadas.

## Features
- **Texto de Formato Longo:** Ideal para treinar modelos de linguagem na compreensão de contexto e dependências de longo alcance.
- **Diversidade de Gêneros:** Embora com viés, inclui uma variedade de gêneros de ficção, como Romance, Fantasia e Ficção Científica.
- **Grande Escala:** Um dos primeiros datasets de grande escala para pré-treinamento de modelos de PLN.
- **Base para Modelos Fundamentais:** Serviu como base para o desenvolvimento de modelos como BERT, RoBERTa e GPT-N.

## Use Cases
- **Pré-treinamento de Modelos de Linguagem:** Usado para treinar modelos de PLN de grande escala, como BERT, RoBERTa, e modelos da série GPT.
- **Aprendizado de Representações de Sentenças:** Ideal para o aprendizado não supervisionado de codificações de sentenças e parágrafos.
- **Tarefas de Geração de Linguagem:** O texto de formato longo é útil para treinar modelos a gerar narrativas e textos coerentes.
- **Pesquisa em PLN:** Utilizado como benchmark e corpus de treinamento em diversas pesquisas acadêmicas sobre compreensão e geração de linguagem.

## Integration
A forma mais recomendada e acessível de utilizar o BookCorpus é através de repositórios de datasets como o **Hugging Face**, que geralmente oferecem versões pré-processadas e mais limpas (como o BookCorpusOpen ou variantes refinadas).

**Exemplo de Uso com a Biblioteca `datasets` do Hugging Face (Versão `rojagtap/bookcorpus`):**

```python
from datasets import load_dataset

# Carrega o dataset
ds = load_dataset("rojagtap/bookcorpus")

# Acessa o split de treinamento
train_data = ds["train"]

# Exibe o primeiro exemplo
print(train_data[0])
```

**Alternativas:**
- **Kaggle:** Diversas versões refinadas ou parciais do BookCorpus estão disponíveis no Kaggle.
- **Repositórios Não Oficiais:** Devido a questões de direitos autorais e remoção da fonte original (Smashwords), a versão original não está mais disponível diretamente, sendo necessário buscar versões re-hospedadas ou derivadas.

## URL
[https://huggingface.co/datasets/rojagtap/bookcorpus](https://huggingface.co/datasets/rojagtap/bookcorpus)
