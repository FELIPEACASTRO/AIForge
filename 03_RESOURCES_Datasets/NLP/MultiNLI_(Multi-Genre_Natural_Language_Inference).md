# MultiNLI (Multi-Genre Natural Language Inference)

## Description
O **Multi-Genre Natural Language Inference (MultiNLI)** é um corpus de grande escala, crowdsourced, composto por 433 mil pares de frases anotadas com informações de inferência textual (entailment). Ele foi modelado a partir do corpus SNLI, mas se diferencia por abranger uma variedade de **dez gêneros** de texto falado e escrito (como Ficção, Cartas, Discurso Telefônico, Relatório 9/11, etc.). O principal objetivo do MultiNLI é permitir uma avaliação de generalização distinta e mais robusta, conhecida como avaliação **cross-genre**, onde os modelos são testados em um gênero diferente daquele em que foram treinados. O corpus é um recurso fundamental para o desenvolvimento e avaliação de modelos de compreensão de linguagem natural (NLU) e serviu como base para a tarefa compartilhada do RepEval 2017 Workshop [1] [2].

## Statistics
- **Tamanho do Dataset**: 433.000 (433k) pares de frases anotadas.
- **Tamanho do Arquivo**: 227 MB (ZIP).
- **Versão Principal**: 1.0 (Versão 0.9 também existe, mas difere apenas nos campos `pairID` e `promptID`).
- **Divisão (Aproximada)**:
    - Treinamento: ~392.7k pares
    - Desenvolvimento (Matched): ~9.8k pares
    - Desenvolvimento (Mismatched): ~9.8k pares
    - Teste (Matched): ~9.8k pares (disponível via Kaggle/GLUE)
    - Teste (Mismatched): ~9.8k pares (disponível via Kaggle/GLUE)

## Features
- **Multi-Gênero**: Inclui 10 gêneros distintos de texto, o que o torna mais desafiador e representativo da linguagem real do que o SNLI.
- **Inferência Textual**: Cada par de frases (Premissa e Hipótese) é rotulado com uma das três relações: **entailment** (implicação), **contradiction** (contradição) ou **neutral** (neutro).
- **Avaliação Cross-Genre**: O conjunto de desenvolvimento e teste é dividido em duas partes: *Matched* (o mesmo gênero do conjunto de treinamento) e *Mismatched* (gêneros não vistos no treinamento), permitindo uma avaliação da capacidade de generalização do modelo.
- **Formato**: Distribuído em arquivos ZIP contendo os dados nos formatos JSON Lines (.jsonl) e texto separado por tabulação (.txt).
- **Licença**: A licença é detalhada no artigo de descrição dos dados [1].

## Use Cases
- **Treinamento de Modelos NLU**: Principalmente para treinar e ajustar modelos de Inferência de Linguagem Natural (NLI), como BERT, RoBERTa e LLMs.
- **Avaliação de Generalização**: Utilizado para testar a capacidade de um modelo de generalizar o entendimento de inferência textual para novos domínios (gêneros) de texto.
- **Pesquisa em NLU**: Serve como um benchmark fundamental para a pesquisa em Compreensão de Linguagem Natural, especialmente em tarefas de raciocínio e inferência.
- **Transfer Learning**: Usado como um dataset de pré-treinamento ou *fine-tuning* em tarefas relacionadas à semântica e ao raciocínio textual.

## Integration
O dataset MultiNLI (versão 1.0) pode ser baixado diretamente do site oficial da NYU.
1. **Download**: Baixe o arquivo ZIP (227MB) através do link fornecido na seção URL.
2. **Descompactação**: O arquivo contém os dados nos formatos `.jsonl` e `.txt`.
3. **Uso**: Para a maioria das aplicações modernas, o dataset é facilmente acessível através de bibliotecas como **Hugging Face Datasets** ou **TensorFlow Datasets (TFDS)**, que gerenciam o download, a divisão e o pré-processamento automaticamente.

**Exemplo de Integração (Hugging Face Datasets):**
```python
from datasets import load_dataset

# Carrega o dataset MultiNLI
# O 'mismatch' é a parte mais desafiadora para avaliação de generalização
dataset = load_dataset("multi_nli", split="validation_mismatched")

# Exibe um exemplo
print(dataset[0])
```

## URL
[https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)
