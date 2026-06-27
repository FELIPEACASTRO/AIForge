# CLIP (Contrastive Language-Image Pre-Training)

## Description

Uma rede neural multimodal desenvolvida pela OpenAI que aprende conceitos visuais a partir de supervisão em linguagem natural. Sua proposta de valor única reside na capacidade de conectar imagens e texto em um espaço de *embedding* compartilhado, permitindo a transferência de conhecimento de forma eficiente e a realização de tarefas de visão *zero-shot* (sem a necessidade de *fine-tuning* para novas classes). Foi treinado para prever qual legenda de um conjunto de 32.768 legendas aleatórias foi pareada com uma determinada imagem, usando um *dataset* massivo de 400 milhões de pares imagem-texto.

## Statistics

Desenvolvido pela OpenAI. Treinado em um *dataset* de 400 milhões de pares imagem-texto. Arquitetura de *dual-encoder* (codificador de imagem e codificador de texto). Modelos de imagem incluem ResNet-50 e Vision Transformer (ViT-L/14). Alcançou 76.2% de acurácia top-1 em classificação *zero-shot* no ImageNet.

## Features

Classificação de imagem *zero-shot*; Busca e recuperação multimodal (imagem-texto e texto-imagem); Geração de imagem condicionada a texto (como base para modelos como DALL-E); Transferência de aprendizado para diversas tarefas de visão.

## Use Cases

Busca e recuperação de conteúdo multimodal em larga escala; Classificação de imagem em domínios não vistos (zero-shot); Filtragem e moderação de conteúdo; Geração de arte e imagens (como componente fundamental em modelos generativos).

## Integration

O CLIP é facilmente acessível através da biblioteca `transformers` do Hugging Face. A integração envolve o uso de um codificador de imagem e um codificador de texto para gerar *embeddings* que podem ser comparados para determinar a similaridade.
\n**Exemplo de Classificação Zero-Shot (Python com Hugging Face):**
\n```python
\nfrom transformers import CLIPProcessor, CLIPModel
\nfrom PIL import Image
\n\n# Carregar modelo e processador
\nmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
\nprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
\n\n# Imagem e classes candidatas
\nimage = Image.open("path/to/your/image.jpg")
\ntexts = ["uma foto de um gato", "uma foto de um cachorro", "uma foto de um pássaro"]
\n\n# Processar e obter previsões
\ninputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
\noutputs = model(**inputs)
\nlogits_per_image = outputs.logits_per_image # este é o logit de similaridade
\nprobs = logits_per_image.softmax(dim=1) # converter para probabilidades
\n\n# Exibir resultado
\nprint(f"Probabilidades: {probs.tolist()[0]}")
\nprint(f"Classe prevista: {texts[probs.argmax().item()]}")
\n```

## URL

https://openai.com/index/clip/