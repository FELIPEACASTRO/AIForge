# Self-Supervised Learning: SimCLR, MoCo, BYOL

## Description

SimCLR, MoCo e BYOL são métodos de Aprendizagem Auto-Supervisionada (SSL) que revolucionaram a forma como os modelos de Visão Computacional aprendem representações visuais sem a necessidade de rótulos humanos. Eles se enquadram na categoria de Aprendizagem Contrastiva (SimCLR, MoCo) ou métodos baseados em similaridade (BYOL), com o objetivo comum de aprender um espaço de representação onde as transformações (vistas aumentadas) da mesma imagem (pares positivos) são agrupadas, enquanto as representações de imagens diferentes (pares negativos, exceto no BYOL) são separadas. O sucesso desses modelos reside em sua capacidade de pré-treinar encoders robustos em grandes conjuntos de dados não rotulados, que podem então ser ajustados com eficiência para tarefas downstream com dados rotulados limitados.

## Statistics

**Desempenho (Acurácia Top-1 em ImageNet, Ajuste Fino Linear com ResNet-50):**
*   **SimCLR (v2):** 71.1% (com lote de 4096)
*   **MoCo (v2):** 71.1% (com lote de 256 e fila de 65536)
*   **BYOL:** 74.3% (sem negativos, com normalização de lote)

**Requisitos de Treinamento:**
| Método | Tamanho do Lote (Batch Size) | Uso de Negativos | Mecanismo para Evitar Colapso |
| :--- | :--- | :--- | :--- |
| **SimCLR** | Muito Grande (e.g., 4096) | Sim (outras amostras no lote) | Lote grande e aumentações fortes |
| **MoCo** | Pequeno a Médio (e.g., 256) | Sim (fila de recursos) | Encoder de Momento e Fila de Recursos |
| **BYOL** | Pequeno a Médio (e.g., 256) | Não | Rede Alvo de Momento e Preditor |

**Observação:** O BYOL demonstrou ser mais robusto a diferentes aumentações de dados do que o SimCLR, e sua versão com normalização de grupo (GN) + padronização de peso (WS) alcançou 74.3% de acurácia Top-1, superando a versão com Normalização de Lote (BN) (73.9%).

## Features

**SimCLR (Simple Framework for Contrastive Learning of Visual Representations):**
*   **Arquitetura:** Encoder (ResNet) + Projection Head (MLP não linear).
*   **Mecanismo:** Maximiza a concordância entre duas vistas aumentadas da mesma imagem (par positivo) e minimiza a concordância com todas as outras imagens no lote (pares negativos).
*   **Requisito Chave:** Requer um **tamanho de lote muito grande** (e.g., 4096) para fornecer um número suficiente de amostras negativas de alta qualidade.
*   **Função de Perda:** NT-Xent (Normalized Temperature-Scaled Cross-Entropy Loss).

**MoCo (Momentum Contrast):**
*   **Arquitetura:** Encoder de Consulta (Query Encoder) e Encoder de Chave (Key Encoder) de Momento + Fila de Recursos (Feature Queue).
*   **Mecanismo:** Trata a Aprendizagem Contrastiva como uma pesquisa de dicionário. O Encoder de Chave é atualizado por uma Média Móvel Exponencial (EMA) suave do Encoder de Consulta, e a Fila de Recursos armazena representações de chaves passadas, permitindo um **dicionário de negativos grande e consistente** sem a necessidade de um lote grande.
*   **Requisito Chave:** Permite um tamanho de lote menor e é mais eficiente em termos de memória do que o SimCLR.

**BYOL (Bootstrap Your Own Latent):**
*   **Arquitetura:** Rede Online (Estudante) e Rede Alvo (Professor) de Momento + Preditor.
*   **Mecanismo:** Treina a Rede Online para prever a representação da Rede Alvo de uma vista aumentada diferente da mesma imagem. **Não utiliza pares negativos**. A Rede Alvo é uma EMA da Rede Online, fornecendo um alvo estável para a aprendizagem.
*   **Proposta de Valor Única:** Evita o colapso (solução trivial) sem o uso de negativos, o que o torna mais robusto a diferentes aumentações de dados.

## Use Cases

O pré-treinamento com SimCLR, MoCo e BYOL é amplamente utilizado para gerar representações robustas em domínios com escassez de dados rotulados.

*   **Visão Computacional Geral:** Pré-treinamento de encoders para tarefas de classificação, detecção de objetos e segmentação semântica em conjuntos de dados como ImageNet, COCO e Pascal VOC.
*   **Imagiologia Médica:** Treinamento em grandes volumes de imagens médicas não rotuladas (e.g., raios-X, ressonâncias magnéticas) para aprender características relevantes, seguido de ajuste fino em tarefas específicas como detecção de tumores ou classificação de doenças.
*   **Análise de Vídeo:** Adaptação de MoCo e BYOL para aprender representações de vídeo, onde o contraste é feito entre diferentes quadros do mesmo vídeo (pares positivos) ou entre vídeos diferentes (negativos).
*   **Sistemas de Recomendação:** Uso das representações aprendidas para codificar itens (e.g., produtos, filmes) ou interações de usuários, melhorando a qualidade das recomendações em plataformas como Spotify e Pinterest.
*   **Robótica e Veículos Autônomos:** Aprendizagem de representações visuais para compreensão de cenas e detecção de objetos em tempo real, onde a coleta de dados rotulados é cara e perigosa.

## Integration

A integração desses modelos geralmente envolve duas etapas: pré-treinamento e ajuste fino (fine-tuning).

**Pré-treinamento (Exemplo com PyTorch - Conceitual):**
```python
import torch
import torch.nn as nn
from torchvision import models

# 1. Definir o Encoder (e.g., ResNet-50)
encoder = models.resnet50(pretrained=False)
# Remover a camada de classificação final
encoder.fc = nn.Identity() 

# 2. Definir a Arquitetura Específica (SimCLR, MoCo, ou BYOL)
# Para SimCLR: Adicionar Projection Head (MLP)
projection_head = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 128) # Espaço de projeção
)

# Para MoCo/BYOL: Definir a lógica de Momentum Update
# Exemplo de atualização de momento para MoCo/BYOL
@torch.no_grad()
def update_momentum_encoder(online_net, target_net, m=0.999):
    for param_q, param_k in zip(online_net.parameters(), target_net.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

# 3. Treinar com a Loss Function apropriada (NT-Xent para SimCLR/MoCo, MSE para BYOL)
# O código de treinamento real envolve a geração de vistas aumentadas (t1, t2) e o cálculo da perda.
```

**Ajuste Fino (Fine-Tuning):**
Após o pré-treinamento, o `encoder` (ResNet sem a cabeça de projeção) é usado como extrator de recursos e uma nova camada de classificação é adicionada para a tarefa downstream.

```python
# Carregar o encoder pré-treinado
pretrained_encoder = load_pretrained_weights(encoder_path)

# Adicionar uma nova camada de classificação
classifier = nn.Sequential(
    pretrained_encoder,
    nn.Linear(2048, num_classes) # num_classes é o número de classes da tarefa downstream
)

# Treinar o classificador na tarefa downstream (e.g., CIFAR-10)
# ... (código de treinamento supervisionado padrão)
```

## URL

SimCLR: https://arxiv.org/abs/2002.05709 | MoCo: https://arxiv.org/abs/1911.05722 | BYOL: https://arxiv.org/abs/2006.07733