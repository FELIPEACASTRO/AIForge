# Open Source Computer Vision Projects - Object Detection and Segmentation (YOLO & Detectron2)

## Description

**Ultralytics YOLO (You Only Look Once)** é uma família de modelos de detecção de objetos e segmentação de código aberto, conhecida por sua **velocidade e precisão em tempo real**. O projeto, liderado pela Ultralytics, evoluiu para uma plataforma unificada que suporta múltiplas tarefas de visão computacional, incluindo detecção, segmentação de instância, classificação e estimativa de pose. Sua principal proposta de valor é fornecer uma solução de IA de visão de ponta, fácil de usar e de alto desempenho, adequada para implantação em produção e dispositivos de borda. A arquitetura é otimizada para equilibrar velocidade e precisão, tornando-o o padrão ouro para aplicações que exigem inferência rápida.

**Detectron2** é a plataforma de próxima geração do Facebook AI Research (FAIR) para detecção de objetos e tarefas de reconhecimento visual. É o sucessor do Detectron e do maskrcnn-benchmark, oferecendo uma estrutura modular e flexível construída em PyTorch. Sua proposta de valor reside em sua **flexibilidade e capacidade de suportar algoritmos de última geração**, como Mask R-CNN, Cascade R-CNN, e PointRend, tornando-o a escolha preferida para pesquisa e aplicações que exigem alta precisão e segmentação em nível de pixel. O Detectron2 é uma ferramenta robusta para quem deseja implementar ou estender modelos complexos de segmentação de instância e panóptica.

## Statistics

**Ultralytics YOLO (Exemplo YOLOv9):**
*   **GitHub Stars:** O repositório Ultralytics (que hospeda o YOLO) possui mais de 48.4k estrelas.
*   **Desempenho (YOLOv9):** Modelos como o YOLOv9-C alcançam 55.6% de mAP (COCO) com uma latência de 24.6 ms (em GPU A100), demonstrando um excelente equilíbrio entre precisão e velocidade.

**Detectron2:**
*   **GitHub Stars:** O repositório Detectron2 possui mais de 33.6k estrelas.
*   **Desempenho (Exemplo Mask R-CNN):** Implementações do Mask R-CNN com backbone ResNet-50-FPN podem atingir um mAP (segmentação) de cerca de 38-40% no COCO, dependendo da configuração. A velocidade é geralmente mais lenta que o YOLO, focando na precisão e na segmentação em nível de pixel.
*   **Linguagens:** Principalmente Python (94.0%), com Cuda (3.2%) e C++ (2.3%).

## Features

**Ultralytics YOLO:**
*   **Plataforma Unificada:** Suporta detecção de objetos, segmentação de instância, classificação, estimativa de pose e rastreamento.
*   **Alto Desempenho:** Modelos otimizados para velocidade e precisão (YOLOv8, YOLOv9).
*   **Fácil de Usar:** Interface Python e CLI simples para treinamento, validação e inferência.
*   **Exportação Flexível:** Suporte para exportação para formatos de implantação como ONNX, OpenVINO, TensorRT e TFLite.
*   **Arquitetura Inovadora (YOLOv9):** Introdução de PGI (Programmable Gradient Information) e GELAN (Generalized Efficient Layer Aggregation Network) para melhor desempenho.

**Detectron2:**
*   **Modularidade:** Design flexível que permite a fácil implementação de novos algoritmos de visão.
*   **Suporte a Modelos SOTA:** Inclui implementações de última geração como Mask R-CNN, Cascade R-CNN, PointRend, e ViTDet.
*   **Segmentação de Instância e Panóptica:** Capacidade robusta de segmentação em nível de pixel.
*   **Baseado em PyTorch:** Integração profunda com o ecossistema PyTorch.
*   **Extensibilidade:** Projetado para ser uma biblioteca de pesquisa, permitindo que os usuários construam novos projetos sobre sua base.

## Use Cases

**Ultralytics YOLO:**
*   **Veículos Autônomos:** Detecção de pedestres, veículos e sinais de trânsito em tempo real.
*   **Vigilância por Vídeo:** Rastreamento e contagem de objetos ou pessoas em feeds de vídeo ao vivo.
*   **Análise de Varejo:** Monitoramento de prateleiras, análise de fluxo de clientes e detecção de checkout sem atrito.
*   **Robótica:** Navegação e interação com o ambiente em tempo real.

**Detectron2:**
*   **Diagnóstico Médico:** Segmentação de tumores, lesões ou órgãos em imagens médicas (MRI, raios-X) para análise precisa.
*   **Controle de Qualidade na Fabricação:** Detecção de defeitos em linhas de montagem com alta precisão em nível de pixel.
*   **Análise de Imagens de Satélite/Aéreas:** Segmentação de edifícios, estradas e áreas de cultivo.
*   **Edição de Vídeo (Rotoscopia):** Criação de máscaras de segmentação precisas para objetos em vídeo.

## Integration

**Ultralytics YOLO (Python):**
A integração é direta através do SDK Python. O exemplo a seguir demonstra a inferência para detecção de objetos:

```python
from ultralytics import YOLO

# Carregar um modelo pré-treinado (YOLOv8n para detecção)
model = YOLO('yolov8n.pt')

# Realizar inferência em uma imagem
results = model('path/to/image.jpg')

# Para segmentação, use um modelo de segmentação
# model_seg = YOLO('yolov8n-seg.pt')
# results_seg = model_seg('path/to/image.jpg')

# Exibir os resultados (caixas delimitadoras, máscaras, etc.)
for r in results:
    print(r.boxes)      # Caixas delimitadoras
    if r.masks is not None:
        print(r.masks)  # Máscaras de segmentação
```

**Detectron2 (Python):**
A inferência com modelos pré-treinados é realizada usando a classe `DefaultPredictor`.

```python
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# 1. Configuração
cfg = get_cfg()
# Adicionar configurações para o modelo (ex: Mask R-CNN)
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Limite de confiança
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.DEVICE = "cpu" # ou "cuda"

# 2. Criar o preditor
predictor = DefaultPredictor(cfg)

# 3. Carregar a imagem
im = cv2.imread("path/to/image.jpg")

# 4. Realizar a inferência
outputs = predictor(im)

# 5. Visualizar os resultados (opcional)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])
```

## URL

**Ultralytics YOLO:** https://github.com/ultralytics/ultralytics
**Detectron2:** https://github.com/facebookresearch/detectron2