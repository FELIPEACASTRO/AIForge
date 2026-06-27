# CLIP (Contrastive Language-Image Pre-training)

## Description

CLIP (Contrastive Language-Image Pre-training) é uma rede neural desenvolvida pela OpenAI que estabelece uma conexão semântica entre visão e linguagem em um espaço de *embedding* compartilhado. Treinado em 400 milhões de pares (imagem, texto), sua proposta de valor única é a capacidade de realizar **Aprendizagem Zero-Shot (Zero-Shot Learning - ZSL)**, permitindo que o modelo generalize para categorias de objetos não vistas durante o treinamento, utilizando apenas descrições de linguagem natural. Isso elimina a necessidade de *fine-tuning* para novas tarefas de classificação de imagens.

## Statistics

O modelo CLIP ViT-L/14 atinge cerca de **75% de acurácia** em classificação zero-shot no ImageNet. Foi treinado em **400 milhões de pares (imagem, texto)**. Demonstrou ser competitivo com modelos totalmente supervisionados em 30 conjuntos de dados de reconhecimento de imagem.

## Features

Embedding Multimodal (visão e texto); Classificação Zero-Shot por linguagem natural; Busca por Imagem e Texto; Robustez a ruídos e variações em comparação com modelos puramente supervisionados.

## Use Cases

Classificação de Imagens Dinâmica (categorias definidas pelo usuário em tempo real); Busca Semântica de Imagens (entendimento de contexto e semântica da consulta de texto); Filtragem de Conteúdo (identificação de conteúdo específico usando descrições de texto); Geoprocessamento/GIS (classificação de imagens de satélite ou aéreas).

## Integration

A integração é tipicamente feita usando a biblioteca `transformers` do Hugging Face ou a implementação original da OpenAI.

**Exemplo de Classificação Zero-Shot (Python/PyTorch):**

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Carregar o modelo e o processador
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Preparar a imagem e os rótulos de texto
# Suponha que 'image' é um objeto PIL.Image
# image = Image.open("caminho/para/sua/imagem.jpg")
# Exemplo de imagem simulada
image = Image.new('RGB', (200, 200), color = 'red') 

candidate_labels = ["uma foto de um gato", "uma foto de um cachorro", "uma foto de um carro"]
text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True)
image_inputs = processor(images=image, return_tensors="pt")

# 3. Calcular as similaridades
with torch.no_grad():
    outputs = model(**image_inputs, **text_inputs)
    logits_per_image = outputs.logits_per_image # similaridade imagem-texto
    probs = logits_per_image.softmax(dim=1) # probabilidades

# 4. Exibir o resultado
print("Probabilidades de Classificação Zero-Shot:")
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"- {label}: {prob:.4f}")
```

## URL

https://openai.com/index/clip/