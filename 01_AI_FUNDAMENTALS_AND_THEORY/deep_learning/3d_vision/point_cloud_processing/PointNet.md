# PointNet

## Description

Primeira arquitetura de rede neural profunda a consumir diretamente nuvens de pontos (point clouds) desordenadas, garantindo invariância à permutação e robustez a transformações rígidas. Oferece uma arquitetura unificada para tarefas de classificação e segmentação 3D. Principais recursos: **Invariância à Permutação:** Usa uma função de agregação simétrica (max pooling) para lidar com a natureza desordenada da nuvem de pontos. **T-Net (Transformation Network):** Uma mini-rede para aprender transformações afins (affine transformations) para alinhar a nuvem de pontos de entrada e as características intermediárias, tornando o modelo robusto a transformações rígidas. **Extração de Características Globais:** Gera um vetor de características global (global feature vector) para toda a nuvem de pontos.

## Statistics

No paper original (2017), alcançou 89.2% de acurácia em classificação de objetos no ModelNet40 e 85.1% de acurácia de instância na segmentação de partes no ShapeNet. É a base para mais de 22.000 citações (em 2024), indicando seu impacto seminal.

## Features

**Invariância à Permutação:** Usa uma função de agregação simétrica (max pooling) para lidar com a natureza desordenada da nuvem de pontos. **T-Net (Transformation Network):** Uma mini-rede para aprender transformações afins (affine transformations) para alinhar a nuvem de pontos de entrada e as características intermediárias, tornando o modelo robusto a transformações rígidas. **Extração de Características Globais:** Gera um vetor de características global (global feature vector) para toda a nuvem de pontos.

## Use Cases

Classificação de objetos 3D (ex: ModelNet), Segmentação Semântica de Cenas (ex: S3DIS), Segmentação de Partes de Objetos (ex: ShapeNet), Reconhecimento de Primitivas Geométricas.

## Integration

A implementação é baseada em MLP por ponto e max-pooling.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet, self).__init__()
        # T-Net de entrada (simplificado)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Max pooling para obter a característica global
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # Classificação
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
```

## URL

https://arxiv.org/abs/1612.00593