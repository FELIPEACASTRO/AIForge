# The Pile (EleutherAI)

## Description
O The Pile é um vasto e diversificado conjunto de dados de modelagem de linguagem de código aberto, criado pela EleutherAI. Com um tamanho aproximado de 825 GiB (ou 886 GB), ele é composto por 22 subconjuntos menores e de alta qualidade, abrangendo uma ampla gama de domínios. Sua principal característica é a diversidade, que visa melhorar o conhecimento geral e a capacidade de generalização dos Large Language Models (LLMs) treinados com ele. O dataset foi lançado em 2020, mas continua sendo uma referência fundamental na pesquisa de LLMs. Uma versão mais recente e focada em licenças abertas, chamada "The Common Pile v0.1" (8 TB), foi anunciada em 2025.

## Statistics
**Tamanho:** Aproximadamente 825 GiB (ou 886 GB).
**Composição:** 22 subconjuntos de dados distintos.
**Versão Principal:** A versão original foi lançada em 2020.
**Versão Relacionada Recente:** "The Common Pile v0.1" (8 TB), anunciada em junho de 2025, é uma versão focada em conteúdo de domínio público e licenças abertas.
**Amostras:** Não há um número total de documentos facilmente disponível, mas o dataset é composto por mais de 400 bilhões de tokens.

## Features
Diversidade de Domínios: Inclui 22 subconjuntos de dados, como código, artigos científicos (arXiv, PubMed Central), livros, conversas de chat (Ubuntu IRC), documentos legais (FreeLaw) e páginas da web (Pile-CC). Qualidade Curada: Os subconjuntos foram cuidadosamente selecionados para garantir alta qualidade e relevância para o treinamento de LLMs. Open Source: Disponível publicamente para a comunidade de pesquisa. Formato: Os dados são fornecidos em formato `jsonlines` comprimido com `zstandard`.

## Use Cases
Treinamento de Large Language Models (LLMs) de propósito geral, como GPT-J e GPT-NeoX. Avaliação da capacidade de generalização e conhecimento de mundo de modelos de linguagem (usando o Pile BPB - Bits Per Byte). Pesquisa em diversidade de dados e seu impacto no desempenho de modelos de linguagem. Fine-tuning de modelos para tarefas específicas em domínios como ciência, medicina e programação.

## Integration
O dataset The Pile pode ser acessado e baixado de várias fontes. A fonte primária de download era o The Eye, mas o Hugging Face é o método de integração mais recomendado e atualizado para uso em projetos de Machine Learning.

**Via Hugging Face Datasets (Recomendado):**
```python
from datasets import load_dataset

# O dataset completo é muito grande, o Hugging Face geralmente requer
# a especificação de um subconjunto ou o uso de streaming.
# Para carregar o dataset completo (pode ser inviável devido ao tamanho):
# dataset = load_dataset("EleutherAI/pile")

# Para carregar um subconjunto específico (ex: Pile-CC):
# dataset = load_dataset("EleutherAI/pile", "pile_cc")

# Para carregar o dataset em modo streaming (recomendado para datasets grandes):
# dataset = load_dataset("EleutherAI/pile", streaming=True)
```

**Download Direto:**
O download direto dos arquivos `.jsonl.zst` pode ser feito através de mirrors como o Academic Torrents ou repositórios da comunidade, já que o link original do The Eye pode estar inativo. Recomenda-se verificar o repositório oficial do GitHub para links de download atualizados.

## URL
[https://pile.eleuther.ai/](https://pile.eleuther.ai/)
