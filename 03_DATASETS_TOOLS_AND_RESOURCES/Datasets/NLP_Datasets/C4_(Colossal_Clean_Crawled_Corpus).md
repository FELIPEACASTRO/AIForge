# C4 (Colossal Clean Crawled Corpus)

## Description
O **C4 (Colossal Clean Crawled Corpus)** é um dataset massivo de texto em inglês, limpo e rastreado da web, criado pelo Google para o pré-treinamento do modelo T5 (Text-to-Text Transfer Transformer). Ele é derivado de um *snapshot* de abril de 2019 do Common Crawl, aplicando-se uma série de filtros rigorosos para remover conteúdo de baixa qualidade, duplicações, código-fonte, frases incompletas e conteúdo não-inglês. O objetivo principal é fornecer um corpus de alta qualidade para o treinamento de modelos de linguagem de grande escala (LLMs). Embora a versão original seja apenas textual, a pesquisa recente (2023-2025) destaca a importância de sua extensão multimodal, o **Multimodal C4 (mmc4)**, que intercala milhões de imagens com o texto, expandindo seu uso para modelos multimodais.

## Statistics
**Versão C4 (Inglês - Padrão):**
- **Tamanho do Dataset:** 806.87 GiB
- **Amostras (Documentos):** 364.613.570 (split de treino) + 364.724 (split de validação)
- **Versão TFDS:** 3.1.0 (mais recente)

**Versão Multilíngue (mC4):**
- **Tamanho do Dataset:** 38.49 TiB
- **Idiomas:** 101

**Versão Multimodal (mmc4 - 2023):**
- **mmc4 (Completo):** 571 milhões de imagens, 101.2 milhões de documentos, 43 bilhões de tokens.
- **mmc4-ff (Fewer Faces):** 375 milhões de imagens, 77.7 milhões de documentos, 33 bilhões de tokens.

## Features
- **Limpeza Rigorosa:** Aplicação de filtros para remover conteúdo de baixa qualidade, código, frases curtas e duplicatas, resultando em um corpus de texto de alta fidelidade.
- **Base no Common Crawl:** Derivado de um *snapshot* de 2019 do Common Crawl, uma fonte massiva de dados da web.
- **Foco em Texto em Inglês:** A versão original é focada em conteúdo em inglês.
- **Versão Multilíngue (mC4):** Uma configuração que abrange 101 idiomas e é gerada a partir de 86 *dumps* do Common Crawl.
- **Extensão Multimodal (mmc4):** Versão mais recente (2023) que adiciona 571 milhões de imagens alinhadas com o texto, expandindo as capacidades do dataset.

## Use Cases
- **Pré-treinamento de Modelos de Linguagem de Grande Escala (LLMs):** Foi o dataset fundamental para o treinamento do modelo T5 e é amplamente utilizado como *baseline* para outros LLMs.
- **Geração de Texto:** Treinamento de modelos para tarefas de geração de texto de alta qualidade.
- **Modelos Multimodais (com mmc4):** Treinamento de modelos que integram texto e imagem, como modelos de *vision-language* e *multimodal large language models* (MLLMs).
- **Pesquisa em Processamento de Linguagem Natural (PLN):** Serve como um corpus limpo e massivo para diversas tarefas de pesquisa em PLN.

## Integration
O dataset C4 não é fornecido diretamente para download pelo Google devido ao seu tamanho e aos custos de largura de banda. Em vez disso, o método de integração recomendado é a **reprodução** a partir dos dados brutos do Common Crawl, utilizando as ferramentas de código aberto fornecidas pelo projeto T5 e o TensorFlow Datasets (TFDS).

**Método de Integração (TFDS):**
1.  **Instalação:** Instale o `tfds-nightly` com a dependência `c4`: `pip install tfds-nightly[c4]`
2.  **Geração Distribuída:** Devido ao tamanho (~7 TB de dados brutos) e ao tempo de processamento (~335 CPU-dias), é altamente recomendado usar um serviço de processamento distribuído como o **Google Cloud Dataflow** com o Apache Beam, seguindo as instruções detalhadas no repositório T5.
3.  **Acesso via Hugging Face:** Versões pré-processadas e menores (como o `allenai/c4` ou `brando/small-c4-dataset`) estão disponíveis no Hugging Face Datasets para experimentação mais leve.

**Integração Multimodal (mmc4):**
A versão Multimodal C4 (mmc4-ff) está disponível para download direto no Hugging Face Datasets:
- `jmhessel/mmc4-ff` (~218GB)
- `jmhessel/mmc4-core-ff` (~20GB)

## URL
[https://www.tensorflow.org/datasets/catalog/c4](https://www.tensorflow.org/datasets/catalog/c4)
