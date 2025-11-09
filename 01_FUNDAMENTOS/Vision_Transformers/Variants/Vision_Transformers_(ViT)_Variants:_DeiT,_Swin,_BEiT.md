# Vision Transformers (ViT) Variants: DeiT, Swin, BEiT

## Description

**DeiT (Data-efficient Image Transformers):** São transformadores treinados de forma mais eficiente para classificação de imagens, exigindo muito menos dados e poder computacional do que o ViT original. A principal inovação é a técnica de **destilação** (distillation), onde um modelo Transformer é treinado usando um modelo de rede neural convolucional (CNN) já treinado (o "professor") para guiar o treinamento do Transformer (o "aluno"). Isso permite que o DeiT atinja um desempenho competitivo com CNNs e ViTs treinados em grandes conjuntos de dados, usando apenas o ImageNet-1k. **Swin Transformer (Shifted Window Transformer):** É um Vision Transformer que serve como um **backbone de propósito geral** para visão computacional. A principal inovação é a introdução de uma **arquitetura hierárquica** e um mecanismo de **atenção de janela deslocada** (shifted window attention). Aborda os problemas de escala e complexidade quadrática do ViT original, permitindo que o modelo se adapte a uma ampla gama de tarefas de visão e escale para imagens de alta resolução. **BEiT (Bidirectional Encoder representation from Image Transformers):** É um modelo de representação de visão **auto-supervisionado** que segue o paradigma de pré-treinamento **BERT-style** (Masked Language Modeling) para imagens. A abordagem de pré-treinamento é chamada de **Masked Image Modeling (MIM)**. O modelo reconstrói os "tokens visuais" (visual tokens) de patches de imagem mascarados. Transfere o sucesso do pré-treinamento auto-supervisionado de linguagem (BERT) para a visão.

## Statistics

**DeiT:** O modelo de referência (86M parâmetros) atinge 83.1% de precisão top-1 (single-crop) no ImageNet-1k sem dados externos. Pode ser treinado em um único computador em menos de 3 dias. **Swin Transformer:** Cria representações hierárquicas, com eficiência aprimorada pelo esquema de atenção de janela deslocada. Suporta modelos de visão de até 3 bilhões de parâmetros (Swin Transformer V2). **BEiT:** Demonstrou melhorias significativas em tarefas downstream, como classificação de imagens e segmentação semântica, após o pré-treinamento MIM. Utiliza grandes conjuntos de dados de imagens não rotuladas para o pré-treinamento.

## Features

**DeiT:** Arquitetura Transformer padrão, introdução do **token de destilação** (distillation token), foco na **eficiência de dados** e **eficiência de treinamento**. **Swin Transformer:** **Atenção de Janela Deslocada** que limita a atenção a janelas locais, mas permite a comunicação entre janelas em camadas sucessivas. **Arquitetura Hierárquica** que gera mapas de recursos em diferentes resoluções. **Versatilidade** como backbone para diversas tarefas. **BEiT:** **Pré-treinamento MIM** (Masked Image Modeling) para reconstrução de patches mascarados. Uso de **Tokens Visuais** discretos para aplicar o paradigma BERT à visão. Arquitetura Transformer Encoder padrão.

## Use Cases

**DeiT:** Classificação de Imagens em cenários com recursos limitados de dados ou computação. Base para outros trabalhos que exploram a destilação de conhecimento em Transformers de Visão. **Swin Transformer:** Detecção de Objetos e Segmentação de Instâncias. Segmentação Semântica. Modelos de visão de grande escala. **BEiT:** Pré-treinamento de modelos de visão para diversas tarefas downstream. Classificação de Imagens. Segmentação Semântica (com BEiT V2). Aplicações em diagnóstico de câncer.

## Integration

**DeiT:** Disponível no **Hugging Face Transformers** (e.g., `transformers.DeiTModel`). Implementações em **PyTorch** e **TensorFlow/Keras**. **Swin Transformer:** Disponível no **Hugging Face Transformers**. Implementações oficiais em **PyTorch**. **BEiT:** Disponível no **Hugging Face Transformers**. Implementações oficiais em **PyTorch**.

## URL

DeiT: https://github.com/facebookresearch/deit | https://arxiv.org/abs/2012.12877 | Swin: https://github.com/microsoft/Swin-Transformer | https://arxiv.org/abs/2103.14030 | BEiT: https://www.microsoft.com/en-us/research/publication/beit-bert-pre-training-of-image-transformers/ | https://arxiv.org/abs/2106.08254