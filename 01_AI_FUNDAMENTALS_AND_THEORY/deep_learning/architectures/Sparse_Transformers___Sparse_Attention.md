# Sparse Transformers / Sparse Attention

## Description

Sparse Transformers e a Atenção Esparsa são inovações arquitetônicas que resolvem o gargalo de complexidade quadrática ($O(n^2)$) dos Transformers padrão em relação ao comprimento da sequência ($n$). A técnica principal é a fatoração esparsa da matriz de atenção, que restringe quais tokens interagem, reduzindo a complexidade para $O(n\sqrt{n})$ (Sparse Transformer original) ou $O(n)$ (em variantes como BigBird e Longformer). Isso permite o processamento eficiente de sequências com dezenas de milhares de tokens, o que era inviável com a atenção densa. A atenção esparsa engloba subtipos como a **atenção local** (onde cada token atende apenas a vizinhos próximos) e a **atenção global** (onde tokens especiais atendem a toda a sequência).

## Statistics

- **Sequências Mais Longas:** Permite sequências **10x** mais longas para BERT base e **16x** mais longas para BERT large em pré-treinamento, em comparação com a atenção densa.
- **Computação Mais Rápida:** Aceleração de treinamento de até **6.3x** para BERT base, **5.3x** para BERT large e **6.1x** para GPT2 (DeepSpeed).
- **Melhoria de Inferência:** Até **3.13x** mais rápido na inferência em BERT-Base em comparação com Longformer (DeepSpeed).
- **Complexidade de Memória:** Reduz a pegada de memória para $O(wn)$, onde $w$ é o tamanho da janela de atenção local.

## Features

- **Redução de Complexidade:** Transforma a complexidade de tempo e memória de $O(n^2)$ para $O(n\sqrt{n})$ ou $O(n)$.
- **Processamento de Sequências Longas:** Permite o processamento de sequências com milhares de tokens (até 4096 no Longformer, dezenas de milhares no Sparse Transformer original).
- **Mecanismos de Atenção Flexíveis:** Suporta atenção **local**, **global** e **aleatória**, ou qualquer combinação dessas.
- **Estruturas de Esparsidade:** Implementa estruturas populares como **Fixed** (OpenAI Sparse Transformer), **BigBird** e **BSLongformer**.
- **Otimização de Hardware:** Implementações como DeepSpeed Sparse Attention utilizam kernels esparsos otimizados para GPU, como os desenvolvidos com Triton.

## Use Cases

- **Modelagem Generativa de Sequências Longas:** Previsão de pixels em imagens ou geração de texto longo.
- **Processamento de Documentos Longos:** Análise de documentos extensos, como relatórios financeiros, artigos científicos ou textos legais (ex: Long Document Comprehension).
- **Pré-treinamento de Modelos de Linguagem:** Treinamento mais eficiente de modelos como BERT e GPT2 com sequências de entrada maiores.
- **Visão Computacional:** Processamento de sequências de pixels ou patches muito longas.
- **Análise de Dados Tabulares:** Aplicação de aprendizado por transferência em dados tabulares.

## Integration

A integração mais robusta é através de bibliotecas de otimização de treinamento como o **DeepSpeed Sparse Attention** ou através de modelos pré-implementados em bibliotecas como **Hugging Face Transformers**.

**1. Integração com DeepSpeed (Python/PyTorch):**
O DeepSpeed substitui o módulo de atenção padrão por kernels esparsos otimizados, configurados via arquivo JSON.

```python
# Exemplo de configuração de esparsidade no DeepSpeed (ds_config.json)
{
  "sparse_attention": {
    "enabled": true,
    "block_size": 64,
    "sparsity_structure": "fixed", // "fixed", "bigbird", "bslongformer", ou "variable"
    "local_window_size": 256,
    "num_global_blocks": 1,
    "random_blocks": false
  }
}

# Execução do treinamento com o DeepSpeed launcher
# deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

**2. Integração com Hugging Face (Longformer/BigBird):**
Modelos como Longformer e BigBird já possuem a atenção esparsa implementada internamente.

```python
from transformers import LongformerModel, LongformerTokenizer
import torch

# Inicializa o tokenizador e o modelo Longformer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# Exemplo de texto longo
long_text = "..." * 1000 # Sequência com mais de 512 tokens

# Tokeniza o texto e define a máscara de atenção global (necessária para Longformer)
inputs = tokenizer(long_text, return_tensors='pt', max_length=4096, truncation=True)
global_attention_mask = torch.zeros_like(inputs['input_ids'])
global_attention_mask[:, 0] = 1 # Define o token CLS como global

# Passa a máscara de atenção global para o modelo
outputs = model(**inputs, global_attention_mask=global_attention_mask)
```

## URL

https://openai.com/index/sparse-transformer/ | https://arxiv.org/abs/1904.10509 | https://www.deepspeed.ai/2020/09/08/sparse-attention.html