# Variantes de Transformer Eficientes e de Longo Contexto (Infini-attention, ModernBERT, HMT)

## Description

As variantes de Transformer para 2024-2025 concentram-se em resolver as limitações de eficiência e contexto dos modelos originais. Três arquiteturas notáveis são a **Infini-attention**, o **ModernBERT** e o **Hierarchical Memory Transformer (HMT)**. A Infini-attention, proposta pelo Google, permite um contexto "infinito" ao combinar atenção local mascarada com uma memória compressiva de longo prazo, mantendo a complexidade computacional e de memória limitadas. O ModernBERT é uma modernização do BERT, focada em eficiência e contexto estendido (até 8k tokens), utilizando melhorias como *Rotary Positional Embeddings* (RoPE) e camadas GeGLU. O HMT imita a memória humana, usando um sistema hierárquico para preservar e recuperar informações relevantes em longas sequências, transformando o modelo em um modelo recorrente aumentado por memória. Essas inovações são cruciais para aplicações que exigem compreensão profunda de documentos extensos e manutenção de coerência em conversas longas.

## Statistics

**Infini-attention:** Complexidade de tempo $O(1)$ e complexidade de memória $O(1)$ em relação ao comprimento da sequência (após o primeiro segmento). Permite janelas de contexto de 1 milhão de tokens ou mais. **ModernBERT:** Treinado em 2 trilhões de tokens. Contexto de até 8k tokens. Desempenho 9 pontos percentuais superior aos modelos de contexto longo concorrentes em tarefas de recuperação com ColBERT. **HMT:** Demonstra melhoria consistente na qualidade de geração com contexto longo em várias arquiteturas de Transformer.

## Features

**Infini-attention:** Contexto infinito com complexidade de memória e tempo limitadas. Combina atenção local mascarada e atenção linear de longo prazo com memória compressiva. **ModernBERT:** Eficiência aprimorada, contexto estendido (até 8k tokens), desempenho superior em tarefas de *embedding* e recuperação de contexto longo (com ColBERT). Utiliza RoPE e GeGLU. **HMT:** Processamento de contexto longo através de memória hierárquica. Imita a memorização humana, usando recorrência aumentada por memória para preservar e recuperar tokens relevantes.

## Use Cases

**Infini-attention:** Análise de documentos ultra-longos (contratos, relatórios financeiros, código-fonte extenso), chatbots com memória de conversação ilimitada, e processamento de sequências de dados de séries temporais. **ModernBERT:** Tarefas de *embedding* de texto de alta qualidade, recuperação de informações em grandes bases de dados (*retrieval-augmented generation* - RAG) e compreensão de documentos com contexto estendido. **HMT:** Geração de texto coerente em narrativas longas, sumarização de livros ou artigos científicos, e tarefas de Perguntas e Respostas sobre documentos extensos.

## Integration

A integração é facilitada principalmente através da biblioteca **Hugging Face Transformers**.
**ModernBERT:** Modelos pré-treinados estão disponíveis no Hugging Face Hub (ex: `answerdotai/ModernBERT-base`).
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

inputs = tokenizer("Exemplo de texto para ModernBERT.", return_tensors="pt")
outputs = model(**inputs)
# A saída contém os embeddings do último estado oculto
```
**Infini-attention e HMT:** Implementações em PyTorch estão disponíveis em repositórios do GitHub, que podem ser integradas em arquiteturas de modelos existentes.
**Infini-attention (Exemplo de implementação PyTorch):**
```python
# Exemplo conceitual de uso da camada Infini-attention
from infini_attention_pytorch import InfiniAttention

# Inicializa a camada
infini_attn = InfiniAttention(dim=512, heads=8)

# Aplica a atenção a uma sequência de entrada
output = infini_attn(x) # x é o tensor de entrada
```
A adoção dessas variantes requer a substituição do mecanismo de atenção padrão pela nova arquitetura (Infini-attention) ou o uso de modelos pré-treinados (ModernBERT).

## URL

Infini-attention (Paper): https://arxiv.org/abs/2404.07143; ModernBERT (Hugging Face): https://huggingface.co/blog/modernbert; HMT (Paper): https://arxiv.org/abs/2405.06067