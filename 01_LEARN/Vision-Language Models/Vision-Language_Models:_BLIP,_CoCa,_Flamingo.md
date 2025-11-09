# Vision-Language Models: BLIP, CoCa, Flamingo

## Description

Pesquisa abrangente sobre os modelos de Linguagem-Visão (VLMs) BLIP, CoCa e Flamingo, detalhando suas propostas de valor únicas, arquiteturas, métricas de desempenho, casos de uso e métodos de integração. O BLIP se destaca pela unificação de tarefas de compreensão e geração e pelo método CapFilt para dados ruidosos. O CoCa inova ao unificar os paradigmas de encoder-decoder, dual-encoder e single-encoder através de perdas contrastivas e de legendagem. O Flamingo é notável por seu desempenho de última geração em aprendizado few-shot, utilizando uma arquitetura de fusão de modelos congelados com atenção cruzada.

## Statistics

BLIP: SOTA em Image-Text Retrieval (+2.7% recall@1), Image Captioning (+2.8% CIDEr), VQA (+1.6% VQA score). CoCa: SOTA em múltiplas tarefas, 91.0% top-1 accuracy no ImageNet, 86.3% zero-shot accuracy no ImageNet. Flamingo: 80B parâmetros, SOTA em 16 tarefas few-shot multimodais.

## Features

BLIP: Arquitetura MED, CapFilt, Objetivos de Pré-treinamento Unificados. CoCa: Arquitetura Unificada (Contrastiva + Captioning), Encoder-Decoder Desacoplado, Representações Duplas. Flamingo: Aprendizado Few-Shot, Arquitetura de Fusão Congelada, Mecanismo de Atenção Cruzada (Gated Cross-Attention).

## Use Cases

BLIP: Recuperação de Imagem-Texto, Legenda de Imagem, VQA. CoCa: Classificação de Imagens, Reconhecimento de Vídeo, Recuperação Cross-Modal, Legenda de Imagens. Flamingo: Diálogo Multimodal, Aprendizado Few-Shot, Legenda de Vídeos.

## Integration

BLIP: Integração oficial via LAVIS e Hugging Face Transformers. CoCa: Implementações de código aberto via PyTorch (lucidrains/CoCa-pytorch). Flamingo: Implementação de código aberto via OpenFlamingo. Exemplos de código em Python fornecidos para cada modelo.

## URL

BLIP: https://github.com/salesforce/BLIP; CoCa: https://arxiv.org/abs/2205.01917; Flamingo: https://arxiv.org/abs/2204.14198