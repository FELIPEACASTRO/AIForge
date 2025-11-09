# BERT Variants and Successors (RoBERTa, ALBERT, DeBERTa)

## Description

**RoBERTa (Robustly optimized BERT pretraining approach):** Uma abordagem de pré-treinamento do BERT robustamente otimizada, que demonstrou que o BERT estava sub-treinado. Sua proposta de valor é a melhoria substancial de desempenho através da otimização dos hiperparâmetros de pré-treinamento, remoção da tarefa de Previsão da Próxima Sentença (NSP) e uso de mascaramento dinâmico.

**ALBERT (A Lite BERT for Self-supervised Learning of Language Representations):** Uma versão "Lite" do BERT que foca na redução drástica do número de parâmetros. Sua proposta de valor é permitir o escalonamento para modelos muito maiores com menor custo de memória e tempo de treinamento, mantendo ou superando o desempenho do BERT através de técnicas de redução de parâmetros.

**DeBERTa (Decoding-enhanced BERT with Disentangled Attention):** Uma nova arquitetura que aprimora o BERT e o RoBERTa. Sua proposta de valor é alcançar o estado da arte em várias tarefas de PNL através de um mecanismo de atenção desvinculada (disentangled attention) e um decodificador de máscara aprimorado (enhanced mask decoder).

## Statistics

**RoBERTa:** Lançamento em 2019. Treinado em 160GB de texto. Superou o BERT em GLUE e SQuAD.

**ALBERT:** Lançamento em 2019. Redução de até 89% de parâmetros (ALBERT-base com 12M). ALBERT-xxlarge alcançou o estado da arte em 12 tarefas de PNL, incluindo 89.4 no benchmark RACE.

**DeBERTa:** Lançamento em 2020 (Microsoft Research). Supera consistentemente o RoBERTa-Large, com melhorias de +0.9% em MNLI, +2.3% em SQuAD v2.0 e +3.6% em RACE. DeBERTa V3 estabeleceu novos benchmarks.

## Features

**RoBERTa:** Mascaramento Dinâmico, Remoção da Tarefa NSP, Treinamento com Lotes Maiores (até 8000), Tokenizador BPE de Nível de Byte.

**ALBERT:** Fatoração da Parametrização de Embedding, Compartilhamento de Parâmetros entre Camadas (redução de até 89% de parâmetros), Objetivo de Coerência de Sentença (SOP) em vez de NSP.

**DeBERTa:** Atenção Desvinculada (separa conteúdo e posição), Decodificador de Máscara Aprimorado (EMD) para incorporar posições absolutas no decodificador.

## Use Cases

**RoBERTa:** Classificação de Texto (GLUE), Resposta a Perguntas (SQuAD), Extração de Recursos para tarefas de PNL.

**ALBERT:** Aplicações com restrições de memória (dispositivos móveis, edge computing), Resposta a Perguntas, Compreensão de Leitura (RACE).

**DeBERTa:** Classificação de Texto (MNLI), Resposta a Perguntas (SQuAD), Compreensão de Leitura (RACE), tarefas de PNL que exigem compreensão contextual profunda.

## Integration

**RoBERTa (Hugging Face Transformers):**
```python
from transformers import pipeline
pipeline = pipeline(task="fill-mask", model="FacebookAI/roberta-base")
resultado = pipeline("Plants create <mask> through a process known as photosynthesis.")
```

**ALBERT (Hugging Face Transformers):**
```python
from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')
text = "O ALBERT é uma versão mais leve do BERT."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

**DeBERTa (Hugging Face Transformers - AutoModel):**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base")
text = "DeBERTa é o novo estado da arte em PNL."
inputs = tokenizer(text, return_tensors="pt")
```

## URL

RoBERTa: https://huggingface.co/docs/transformers/en/model_doc/roberta | ALBERT: https://arxiv.org/abs/1909.11942 | DeBERTa: https://arxiv.org/abs/2006.03654