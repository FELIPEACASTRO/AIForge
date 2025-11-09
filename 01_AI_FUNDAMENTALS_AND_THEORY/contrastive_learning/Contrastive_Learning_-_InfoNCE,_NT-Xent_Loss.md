# Contrastive Learning - InfoNCE, NT-Xent Loss

## Description

**Contrastive Learning** é um paradigma de aprendizado auto-supervisionado que treina modelos para aprender representações latentes, garantindo que pares de dados "positivos" (vistas aumentadas do mesmo dado) fiquem próximos no espaço de *embedding*, enquanto pares "negativos" (dados distintos) fiquem distantes. A **InfoNCE (Information Noise Contrastive Estimation)** é uma função de perda fundamental que enquadra o aprendizado contrastivo como um problema de classificação, onde o modelo deve identificar a amostra positiva correta em um conjunto de amostras negativas (ruído). A **NT-Xent (Normalized Temperature-Scaled Cross-Entropy Loss)** é uma variação da InfoNCE, popularizada pelo *framework* SimCLR. A NT-Xent aprimora a InfoNCE ao: 1) Aplicar **normalização L2** aos vetores de *embedding* antes de calcular a similaridade (geralmente similaridade de cosseno), o que força os *embeddings* a residirem em uma esfera unitária. 2) Introduzir um **parâmetro de temperatura ($\tau$)** para escalar os *logits* de similaridade, o que é crucial para controlar a concentração dos *embeddings* e a dificuldade da tarefa contrastiva. A perda é simétrica, calculada para ambas as direções do par positivo, garantindo um treinamento mais robusto. Em essência, a NT-Xent é a InfoNCE aplicada a *embeddings* normalizados e escalados por temperatura, tornando-a a perda *de facto* para muitos métodos de aprendizado auto-supervisionado de última geração.

## Statistics

*   **Desempenho de *State-of-the-Art* (SOTA):** O *framework* SimCLR, que utiliza a NT-Xent, alcançou 76.5% de precisão *top-1* no ImageNet com uma ResNet-50 (4x) na avaliação *linear probing*, superando significativamente os métodos auto-supervisionados anteriores e reduzindo a lacuna para o treinamento supervisionado.
*   **Requisito de *Batch Size*:** O desempenho da NT-Xent é altamente sensível ao tamanho do *batch*. O SimCLR demonstrou que *batches* grandes (por exemplo, 4096 ou 8192) são cruciais para fornecer um número suficiente de amostras negativas de alta qualidade, o que é essencial para o sucesso do aprendizado contrastivo.
*   **Impacto da Temperatura ($\tau$):** O parâmetro de temperatura é vital. O artigo original do SimCLR mostrou que um $\tau$ bem ajustado (tipicamente entre 0.07 e 0.5) é mais importante do que a escolha do otimizador ou do esquema de aumento de dados.
*   **Eficiência de Transferência:** As representações aprendidas com a NT-Xent em tarefas auto-supervisionadas demonstraram transferibilidade superior, superando o treinamento supervisionado em várias tarefas de *downstream* (por exemplo, em 5 de 12 tarefas de transferência no SimCLR).

## Features

*   **Aprendizado Auto-Supervisionado:** Permite o treinamento de codificadores robustos sem a necessidade de rótulos humanos, utilizando transformações de dados (aumentações) para gerar pares positivos.
*   **Discriminação de Instâncias:** Trata cada instância de dados (e suas aumentações) como uma classe única, forçando o modelo a distinguir entre instâncias individuais.
*   **Normalização L2:** Normaliza os vetores de *embedding* para a esfera unitária, o que estabiliza o treinamento e melhora a qualidade das representações.
*   **Parâmetro de Temperatura ($\tau$):** Um hiperparâmetro ajustável que controla a dispersão dos *embeddings* e a importância das amostras negativas difíceis. Valores menores de $\tau$ tornam a distribuição de probabilidade mais nítida, aumentando a penalidade para *embeddings* que não estão perfeitamente alinhados.
*   **Simetria:** A perda é calculada simetricamente para ambas as vistas aumentadas do par positivo, garantindo que ambas as representações sejam igualmente otimizadas.

## Use Cases

*   **Visão Computacional (SimCLR, MoCo):** O caso de uso primário é o aprendizado de representações visuais robustas em grandes conjuntos de dados não rotulados (como ImageNet), que podem ser transferidas para tarefas de *downstream* com poucos dados rotulados (por exemplo, classificação de imagens, detecção de objetos, segmentação).
*   **Processamento de Linguagem Natural (NLP) (BERT-flow, SimCSE):** Usada para aprimorar a qualidade dos *embeddings* de frases. A perda contrastiva ajuda a agrupar frases com significados semelhantes (pares positivos) e a separá-las de frases não relacionadas (pares negativos), resultando em *embeddings* mais semanticamente coerentes.
*   **Sistemas de Recomendação (RecSys):** Aplicada para aprender representações de usuários e itens a partir de dados de interação esparsos. A perda contrastiva é usada para garantir que as representações de itens com os quais um usuário interagiu (pares positivos) sejam mais próximas do *embedding* do usuário do que as representações de itens não interagidos (pares negativos).
*   **Séries Temporais e Dados Multimodais:** Utilizada para aprender representações de dados de séries temporais (por exemplo, dados de sensores, sinais de áudio) ao contrastar diferentes segmentos de tempo do mesmo sinal (positivo) com segmentos de outros sinais (negativo). Também é eficaz em cenários multimodais, contrastando representações de diferentes modalidades (por exemplo, texto e imagem) que se referem ao mesmo conceito.

## Integration

A implementação da NT-Xent Loss geralmente envolve a concatenação dos *embeddings* de duas vistas aumentadas ($z_i$ e $z_j$) em um *batch* de $2N$, o cálculo da matriz de similaridade de cosseno entre todos os pares, a aplicação da escala de temperatura e, finalmente, o cálculo da perda de entropia cruzada (Cross-Entropy) para identificar o par positivo correto.

**Exemplo de Implementação em PyTorch (NTXentLoss):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross-Entropy Loss (NT-Xent)
    A variant of InfoNCE loss used in SimCLR.
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        N = z_i.shape[0]
        
        # 1. Concatenate all embeddings to form a batch of 2N
        z = torch.cat((z_i, z_j), dim=0) # Shape (2N, D)
        
        # 2. Compute similarity matrix (2N, 2N)
        z = F.normalize(z, dim=1)
        sim_matrix = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        
        # 3. Apply temperature scaling
        sim_matrix = sim_matrix / self.temperature
        
        # 4. Create mask to remove the diagonal (self-similarity)
        logits_mask = torch.ones_like(sim_matrix, dtype=torch.bool).fill_diagonal_(False)
        logits = sim_matrix[logits_mask].view(2 * N, -1) # Shape (2N, 2N - 1)
        
        # 5. Create labels for the positive pairs
        pos_targets = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0)
        labels = (pos_targets - (pos_targets > torch.arange(2 * N))).long()
        
        # 6. Compute the loss and average
        loss = self.criterion(logits, labels)
        loss = loss / (2 * N)
        
        return loss
```

## URL

https://arxiv.org/abs/2002.05709