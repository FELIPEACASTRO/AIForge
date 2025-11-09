# FLOPs-Aware Knowledge Distillation (FAKD) framework e Multistage Knowledge Distillation (MSKD) method

## Description

Pesquisa abrangente sobre a aplicação de Destilação de Conhecimento para implantação de modelos de IA em dispositivos de borda (Edge AI) na agricultura. Os recursos identificados focam em otimização de modelos de Deep Learning para detecção de doenças em plantas e outras aplicações de TinyML em ambientes com recursos limitados.

## Statistics

FAKD: Acurácia no dataset PlantVillage melhorou de 92.77% para 96.55% pós-FAKD, com tradeoff para 90.15% após otimização para implantação em 1MB PSRAM. MSKD: O modelo YOLOR-Light-v2 (distilled) alcançou 60.4% mAP@.5 no dataset PlantDoc, com 20.5M de parâmetros e 20.3GFLOPs, superando o modelo Teacher (YOLOR) em eficiência.

## Features

Dois recursos principais foram identificados: 1) O framework FAKD, que utiliza FLOPs como termo de regularização para otimizar a eficiência computacional de modelos TinyML (MobileNetV2) para hardware restrito (ESP32). 2) O método MSKD, que emprega destilação de conhecimento em múltiplos estágios (backbone, neck, head) para aprimorar a precisão de modelos leves de detecção de objetos (YOLOR-Light-v2) para diagnóstico de doenças em plantas.

## Use Cases

Implantação de modelos eficientes e precisos em dispositivos de baixo consumo de energia em ambientes com recursos limitados (smart agriculture), como equipamentos de campo miniaturizados, sensores IoT e Veículos Aéreos Não Tripulados (VANTs) para diagnóstico de doenças em plantas.

## Integration

O método MSKD possui código de implementação disponível no GitHub (https://github.com/QDH/MSKD) e é baseado em PyTorch. O framework FAKD menciona otimização para implantação em hardware como ESP32, indicando um foco em ferramentas de desenvolvimento TinyML.

## URL

https://ieeexplore.ieee.org/document/10933276/ e https://spj.science.org/doi/10.34133/plantphenomics.0062