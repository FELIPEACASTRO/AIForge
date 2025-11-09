# Segment Anything Medical (MedSAM)

## Description

O **Segment Anything Medical (MedSAM)** é um modelo de fundação (foundation model) pioneiro, projetado para a **segmentação universal de imagens médicas**. Ele foi desenvolvido para preencher a lacuna de generalização dos modelos de segmentação tradicionais, que são tipicamente específicos para uma modalidade ou doença. MedSAM é treinado em um dataset massivo de **1.570.263 pares de imagem-máscara**, cobrindo 10 modalidades de imagem e mais de 30 tipos de câncer. Sua proposta de valor única reside na sua capacidade de oferecer segmentação precisa e eficiente em um amplo espectro de tarefas, demonstrando **melhor precisão e robustez** do que modelos especialistas e superando o SAM original em cenários médicos, especialmente em alvos com limites fracos ou baixo contraste [1] [2].

## Statistics

**Dataset de Treinamento:** 1.570.263 pares de imagem-máscara. **Diversidade de Dados:** 10 modalidades de imagem (CT, MRI, Endoscopia, etc.) e mais de 30 tipos de câncer. **Avaliação:** 86 tarefas de validação interna e 60 tarefas de validação externa, demonstrando robustez e generalização superiores [1]. **Desempenho:** Consistente em superar o SAM original e atingir desempenho igual ou superior a modelos especialistas [1].

## Features

**Segmentação Universal:** Capacidade de segmentar estruturas anatômicas, lesões e regiões patológicas em diversas modalidades de imagem (CT, MRI, Endoscopia, etc.). **Promptable Segmentation:** Utiliza *prompts* (pontos, caixas delimitadoras) para segmentação interativa, oferecendo um equilíbrio entre automação e personalização. **Arquitetura SAM Refinada:** Baseado na arquitetura SAM (codificador de imagem, codificador de prompt e decodificador de máscara), mas fine-tuned para o domínio médico. **MedSAM2 (Extensão):** Versão aprimorada para **segmentação em imagens médicas 3D e vídeos**, permitindo delinear estruturas em varreduras volumétricas com um único clique [3].

## Use Cases

**Diagnóstico e Planejamento de Tratamento:** Segmentação precisa de órgãos, tumores e lesões para planejamento de radioterapia e cirurgia. **Monitoramento de Doenças:** Rastreamento consistente da progressão de doenças, como o crescimento de tumores, em exames sequenciais. **Pesquisa Médica:** Análise de grandes conjuntos de dados de imagens médicas para acelerar a descoberta e validação de novos biomarcadores. **Aplicações Clínicas Diversas:** Segmentação de estruturas em CT, MRI, e inspeção visual de órgãos internos via Endoscopia [1].

## Integration

A integração é realizada tipicamente via PyTorch, seguindo a estrutura do repositório oficial. O modelo requer o download de um *checkpoint* pré-treinado.

**Instalação (MedSAM):**
```bash
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM
pip install -e .
# Baixar o checkpoint do modelo (medsam_vit_b.pth)
```

**Exemplo de Uso (Conceitual - Python/PyTorch):**
```python
import torch
from medsam import SamPredictor, build_medsam
import numpy as np

# 1. Carregar o modelo e o predictor
medsam_checkpoint = "path/to/medsam_vit_b.pth"
model = build_medsam(checkpoint=medsam_checkpoint)
predictor = SamPredictor(model)

# 2. Carregar e processar a imagem (image_data como numpy array)
# predictor.set_image(image_data)

# 3. Definir o prompt (exemplo: um ponto central)
# input_point = np.array([[500, 375]])
# input_label = np.array([1]) # 1 para foreground

# 4. Prever a máscara
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# A máscara (masks[0]) é o resultado da segmentação
```

## URL

https://github.com/bowang-lab/MedSAM