# Funções de Perda para Deep Learning: Perdas Personalizadas e Multi-tarefa

## Description

**Funções de Perda Personalizadas (Custom Losses)** são expressões matemáticas definidas pelo usuário que quantificam a discrepância entre a saída prevista de um modelo de Deep Learning e o valor real, permitindo que o processo de otimização seja ajustado para objetivos específicos que as funções de perda padrão não abordam. Sua proposta de valor única reside na capacidade de **incorporar o conhecimento do domínio e objetivos de negócio diretamente no processo de treinamento**, permitindo a otimização de métricas não-diferenciáveis (como F1-Score ou IoU) e o tratamento de dados específicos (como desequilíbrio de classes ou penalidades assimétricas). **Funções de Perda Multi-tarefa (Multi-task Losses)** são a soma ponderada das perdas individuais de cada tarefa em um modelo de Aprendizado Multi-tarefa (MTL). Sua proposta de valor única é a **capacidade de alavancar o conhecimento compartilhado entre tarefas relacionadas para melhorar o desempenho de todas as tarefas**, sendo o principal desafio o **balanceamento de perdas** para prevenir a transferência negativa e garantir o treinamento equilibrado.

## Statistics

**Ganho de Desempenho em MTL:** Estudos demonstram que o balanceamento de perdas (e.g., GradNorm, Uncertainty Weighting) pode levar a **melhorias de 5% a 15%** no desempenho médio das tarefas (em métricas como mIoU, mAP ou precisão) em comparação com a soma simples de perdas. **Métricas de Balanceamento de Perda:** Em MTL, métricas como a **norma do gradiente por tarefa** e a **incerteza homoscedástica** são usadas internamente para guiar o processo de otimização, atuando como métricas de controle dinâmico. **Distinção Crucial:** A **Função de Perda** (usada para otimização, deve ser diferenciável) é distinta da **Métrica** (usada para avaliação, pode ser não-diferenciável).

## Features

**Perdas Personalizadas:** Incorporação de Restrições (físicas, geométricas), Otimização de Negócio (traduzindo custo de erro em termos diferenciáveis), e Manipulação de Distribuição (e.g., Focal Loss para focar em exemplos difíceis). **Perdas Multi-tarefa:** Balanceamento por Incerteza (Uncertainty Weighting, que aprende pesos dinamicamente com base na incerteza de cada tarefa), Normalização de Gradiente (GradNorm, que ajusta pesos para que os gradientes tenham magnitudes semelhantes), e Cirurgia de Gradiente (PCGrad, que modifica gradientes para evitar conflitos destrutivos entre tarefas).

## Use Cases

**Perdas Personalizadas:** **Visão Computacional** (Uso de Dice Loss ou IoU Loss para otimizar a sobreposição de pixels em segmentação de imagem), **Finanças/Seguros** (Perdas assimétricas que penalizam a subestimação de risco, ou Falso Negativo, muito mais do que a superestimação), e **Processamento de Linguagem Natural (PLN)** (Perdas que incorporam métricas de diversidade ou coerência na geração de texto). **Perdas Multi-tarefa:** **Condução Autônoma** (Treinamento de um único modelo para realizar simultaneamente detecção de objetos, segmentação semântica e estimativa de profundidade), **Análise de Imagem Médica** (Um modelo que segmenta um tumor e classifica o estágio da doença a partir da mesma imagem), e **PLN** (Um modelo que realiza marcação de partes do discurso e reconhecimento de entidades nomeadas simultaneamente).

## Integration

A integração é feita definindo uma função ou classe que herda da classe de perda do framework (e.g., `nn.Module` no PyTorch ou `tf.keras.losses.Loss` no TensorFlow).

**Exemplo 1: Perda Personalizada (Dice Loss em PyTorch)**
```python
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
```

**Exemplo 2: Perda Multi-tarefa (Uncertainty Weighting em PyTorch)**
```python
import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks):
        super(UncertaintyLoss, self).__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigmas[i])
            total_loss += precision * loss + 0.5 * self.log_sigmas[i]
        return total_loss
```

## URL

https://arxiv.org/abs/1705.07113, https://arxiv.org/abs/1711.02257, https://arxiv.org/abs/2001.06782