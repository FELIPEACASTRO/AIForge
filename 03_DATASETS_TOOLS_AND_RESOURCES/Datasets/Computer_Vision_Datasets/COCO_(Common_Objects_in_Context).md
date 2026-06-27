# COCO (Common Objects in Context)

## Description
O COCO (Common Objects in Context) é um dos maiores e mais influentes datasets de visão computacional. Foi projetado para avançar a pesquisa em detecção de objetos, segmentação semântica e de instâncias, reconhecimento em contexto e legendagem de imagens. O dataset é conhecido por suas anotações detalhadas e complexas, incluindo segmentação precisa de instâncias (máscaras de pixel) para cada objeto, além de anotações de "stuff" (coisas sem forma definida, como grama ou céu) e keypoints para pessoas. A versão mais utilizada para competições é a COCO 2017.

## Statistics
**Imagens:** 330K no total (>200K rotuladas). **Instâncias de Objetos:** 1.5 milhão. **Categorias:** 80 categorias de objetos e 91 categorias de "stuff". **Anotações:** 5 legendas por imagem e 250.000 pessoas com keypoints. **Versão 2017 (mais usada):** Imagens de Treinamento (118K/18GB), Validação (5K/1GB), Teste (41K/6GB), Não Rotuladas (123K/19GB). Anotações de Treinamento/Validação (241MB).

## Features
Detecção de objetos em grande escala; Segmentação de instâncias (máscaras de pixel); Segmentação de "stuff" (panoptic); Reconhecimento em contexto; Detecção de keypoints para pessoas; Legendas de imagens (5 por imagem).

## Use Cases
Treinamento e avaliação de modelos de detecção de objetos (por exemplo, YOLO, Faster R-CNN); Segmentação de imagens (semântica e de instâncias); Geração de legendas de imagens (Image Captioning); Detecção de pose humana (Keypoint Detection); Pesquisa em Visão Computacional e Inteligência Artificial.

## Integration
O dataset pode ser baixado diretamente através de links HTTP para os arquivos ZIP das imagens e anotações (versões 2014 e 2017). A forma mais recomendada para download eficiente e manipulação das anotações é através da **COCO API** (disponível no GitHub: https://github.com/cocodataset/cocoapi). A API fornece ferramentas para carregar, analisar e visualizar as anotações, além de facilitar a avaliação de modelos. É sugerido o uso de ferramentas como `gsutil` para evitar o download de grandes arquivos ZIP.

## URL
[https://cocodataset.org/](https://cocodataset.org/)
