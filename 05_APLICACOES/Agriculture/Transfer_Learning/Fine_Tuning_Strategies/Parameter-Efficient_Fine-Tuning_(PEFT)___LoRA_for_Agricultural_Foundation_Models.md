# Parameter-Efficient Fine-Tuning (PEFT) / LoRA for Agricultural Foundation Models

## Description

Estratégias de fine-tuning para modelos de fundação (Foundation Models - FMs) na agricultura, com foco em eficiência e adaptação a tarefas específicas. O método mais proeminente é o **Parameter-Efficient Fine-Tuning (PEFT)**, especialmente o **LoRA (Low-Rank Adaptation)**, que permite a adaptação de modelos pré-treinados massivos (como Large Language Models - LLMs e Visual Large Language Models - VLMs) para tarefas agrícolas com um custo computacional significativamente reduzido. Outras estratégias incluem o **Full Fine-Tuning** (alto custo) e o **Alignment Fine-Tuning** (como RLHF e RLAIF) para alinhar o comportamento do modelo com as necessidades e valores humanos no contexto agrícola. O LoRA é crucial para a democratização da IA na agricultura, permitindo que modelos como AgRoBERTa e WDLM sejam adaptados para tarefas como Question Answering e diagnóstico de doenças de plantas.

## Statistics

**LoRA:** Reduz o número de parâmetros treináveis em ordens de magnitude (por exemplo, 10.000 vezes menos que o ajuste fino completo). **AgRoBERTa (2024):** Utiliza LoRA para Question Answering em extensão agrícola. **WDLM (Wheat Disease Language Model) (2024):** Utiliza LoRA para diagnóstico de doenças do trigo. **Citação:** Artigo de revisão de 2025 (Yin et al.) com 10 citações (em abril de 2025), indicando alta relevância e atualidade.

## Features

**Parameter-Efficient Fine-Tuning (PEFT):** Adaptação de modelos com atualização de um pequeno subconjunto de parâmetros. Inclui **LoRA** (Low-Rank Adaptation) para eficiência. **Full Fine-Tuning:** Atualização de todos os parâmetros para máxima performance. **Alignment Fine-Tuning (RLHF/RLAIF):** Refinamento do comportamento do modelo para alinhamento ético e prático. **Adaptabilidade Multimodal:** Aplicação em modelos de linguagem (LLMs) e modelos visuais (VLMs) para dados de texto e imagem.

## Use Cases

**Diagnóstico de Doenças de Plantas:** Adaptação de VLMs para identificar doenças em culturas específicas (ex: WDLM para trigo). **Question Answering (QA) Agrícola:** Criação de sistemas de resposta a perguntas para extensão agrícola e suporte a decisões de campo (ex: AgRoBERTa). **Classificação de Imagens e Segmentação:** Ajuste de modelos como o Segment Anything Model (SAM) para tarefas como contagem de folhas e segmentação de culturas. **Otimização de Recursos:** Suporte a decisões de irrigação e fertilização baseadas em dados de sensores e imagens.

## Integration

A integração é realizada através de bibliotecas de código aberto que implementam PEFT, como a biblioteca `peft` do Hugging Face. Para modelos como **AgRoBERTa** e **WDLM**, o LoRA é aplicado para injetar matrizes de baixo posto nas camadas do Transformer, permitindo o treinamento eficiente em GPUs de consumo. O processo envolve: 1. Carregar o modelo de fundação pré-treinado. 2. Configurar o adaptador LoRA (rank, alpha, camadas alvo). 3. Treinar apenas os parâmetros do adaptador com um conjunto de dados agrícola específico. 4. Salvar e carregar o adaptador para inferência. (Exemplo de código não disponível no artigo de revisão, mas o método é padrão na comunidade de ML).

## URL

https://www.mdpi.com/2077-0472/15/8/847