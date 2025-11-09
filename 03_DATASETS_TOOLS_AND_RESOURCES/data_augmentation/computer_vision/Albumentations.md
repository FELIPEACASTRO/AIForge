# Albumentations

## Description

Albumentations é uma biblioteca Python rápida e flexível para aumento de dados de imagem, especificamente projetada para projetos de Visão Computacional e Aprendizado Profundo. É amplamente utilizada na indústria e pesquisa devido à sua alta velocidade e rica coleção de transformações. Seu principal diferencial é a otimização de desempenho, sendo mais rápida que a maioria das alternativas em diversas transformações.

## Statistics

Velocidade: Geralmente a mais rápida, com um speedup mediano de 2.64x em comparação com outras bibliotecas (como imgaug, Kornia, torchvision) em transformações de imagem. Throughput: Processa até 10810 imagens/segundo (ex: Brilho) em um único thread de CPU. É a mais rápida em 38 de 48 transformações testadas.

## Features

Ampla gama de transformações (geométricas, de cor, de pixel). Suporte a diferentes tipos de alvos (bounding boxes, máscaras de segmentação, keypoints). API simples para composição de pipelines complexos (A.Compose e A.OneOf). Otimizada para CPU usando OpenCV, resultando em alta velocidade.

## Use Cases

Treinamento de modelos de Visão Computacional para classificação, detecção de objetos e segmentação semântica. Competições de Machine Learning (Kaggle) onde a velocidade de processamento de dados é crucial. Aplicações em produção onde o aumento de dados em tempo real é necessário.

## Integration

Instalação: `pip install -U albumentations`. Exemplo de código:\n```python\nimport albumentations as A\nimport cv2\n\ntransform = A.Compose([\n    A.RandomRotate90(),\n    A.Flip(),\n    A.OneOf([\n        A.MotionBlur(p=.2),\n        A.MedianBlur(blur_limit=3, p=0.1),\n    ], p=0.2),\n    A.HueSaturationValue(p=0.3),\n])\n\nimage = cv2.imread('image.jpg')\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\naugmented_image = transform(image=image)['image']\n```

## URL

https://albumentations.ai/