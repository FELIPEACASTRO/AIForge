# GLUE Benchmark (General Language Understanding Evaluation)

## Description
O **General Language Understanding Evaluation (GLUE) Benchmark** é uma coleção de recursos para treinar, avaliar e analisar sistemas de compreensão de linguagem natural (NLU). O GLUE é composto por um *benchmark* de nove tarefas de NLU baseadas em frases ou pares de frases, selecionadas para cobrir uma gama diversificada de tamanhos de *datasets*, gêneros de texto e graus de dificuldade. Além disso, inclui um *dataset* de diagnóstico para análise detalhada do desempenho do modelo em relação a uma ampla gama de fenômenos linguísticos. O objetivo principal do GLUE é impulsionar a pesquisa no desenvolvimento de sistemas de NLU gerais e robustos, favorecendo modelos que compartilham informações entre tarefas usando técnicas de *transfer learning*. Embora o *benchmark* original seja de 2018, ele continua sendo uma referência fundamental, e *benchmarks* subsequentes como o SuperGLUE (2019) e variações como o Adversarial GLUE (2021) e PrivacyGLUE (2023) demonstram sua relevância contínua e a evolução da pesquisa na área.

## Statistics
O GLUE é uma coleção de *datasets* menores, totalizando centenas de milhares de amostras de treinamento. As estatísticas de treinamento para as nove tarefas originais são:
- **CoLA (Corpus of Linguistic Acceptability):** 8.5k amostras de treinamento.
- **SST-2 (Stanford Sentiment Treebank):** 67k amostras de treinamento.
- **MRPC (Microsoft Research Paraphrase Corpus):** 3.7k amostras de treinamento.
- **STS-B (Semantic Textual Similarity Benchmark):** 7k amostras de treinamento.
- **QQP (Quora Question Pairs):** 364k amostras de treinamento.
- **MNLI (Multi-Genre Natural Language Inference):** 393k amostras de treinamento.
- **QNLI (Question-answering NLI):** 105k amostras de treinamento.
- **RTE (Recognizing Textual Entailment):** 2.5k amostras de treinamento.
- **WNLI (Winograd NLI):** 634 amostras de treinamento.

**Versão:** O *benchmark* original foi introduzido em 2018. Variações e sucessores incluem o **SuperGLUE** (2019) e o **PrivacyGLUE** (2023).

## Features
- **Conjunto de 9 Tarefas de NLU:** Inclui tarefas de classificação de sentença única (CoLA, SST-2), similaridade e paráfrase (MRPC, STS-B, QQP) e inferência textual (MNLI, QNLI, RTE, WNLI).
- **Diagnóstico Linguístico:** Um *dataset* auxiliar projetado para avaliar a compreensão de fenômenos linguísticos específicos.
- **Formato Agnóstico ao Modelo:** Qualquer sistema capaz de processar sentenças e pares de sentenças é elegível.
- **Foco em Transfer Learning:** As tarefas são selecionadas para favorecer modelos que utilizam técnicas de compartilhamento de parâmetros ou *transfer learning* entre tarefas.

## Use Cases
- **Avaliação de Modelos de NLU:** Principalmente usado para medir e comparar o desempenho de modelos de compreensão de linguagem natural, como BERT, RoBERTa e T5, em um conjunto diversificado de tarefas.
- **Transfer Learning e Pré-treinamento:** Utilizado para treinar modelos em múltiplas tarefas simultaneamente (*multi-task learning*) ou para *fine-tuning* de modelos pré-treinados em tarefas específicas.
- **Análise Linguística:** O *dataset* de diagnóstico permite uma análise mais aprofundada das capacidades e deficiências dos modelos em relação a fenômenos linguísticos específicos.
- **Desenvolvimento de Sistemas de NLU Gerais:** Serve como um motor para impulsionar a pesquisa em direção a sistemas de NLU mais robustos e generalizáveis.

## Integration
O *dataset* GLUE pode ser acessado e baixado através de *scripts* fornecidos pela comunidade e pela Hugging Face Datasets. O método mais comum é utilizar a biblioteca `datasets` do Hugging Face ou *scripts* de terceiros, como o `download_glue_data.py` (embora o link original possa estar desatualizado, a funcionalidade é mantida em projetos como o `jiant`).

**Exemplo de uso com Hugging Face Datasets (Python):**
```python
from datasets import load_dataset

# Carrega o dataset GLUE (exemplo: CoLA)
dataset = load_dataset("glue", "cola")

# O dataset é carregado como um objeto DatasetDict
print(dataset)
```
O *benchmark* também é integrado a plataformas como o TensorFlow Datasets (TFDS).

## URL
[https://gluebenchmark.com/](https://gluebenchmark.com/)
