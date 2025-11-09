# Few-Shot Learning (FSL) com PMF+FA e Vision Transformers (ViT)

## Description

O Few-Shot Learning (FSL) é uma abordagem de aprendizado de máquina projetada para treinar modelos de reconhecimento de doenças de plantas com um número muito limitado de exemplos rotulados por classe, o que é crucial para a identificação de **doenças raras em culturas** onde a coleta de dados é escassa.

A técnica mais relevante e recente (2024) identificada é o **PMF+FA** (Pre-training, Meta-learning, Fine-tuning + Feature Attention), que utiliza uma arquitetura de *Vision Transformer* (ViT) para melhorar a representação de características e um módulo de atenção para reduzir a interferência de fundos complexos em imagens de campo.

O FSL, em geral, aborda o problema da escassez de dados, permitindo que os modelos generalizem para novas classes de doenças (as raras) a partir de um conjunto de dados de base (doenças comuns) com apenas algumas amostras de treinamento. Outras abordagens incluem o uso de *Siamese Networks* e *Contrastive Learning* supervisionado.

## Statistics

- **Acurácia:** O modelo PMF+FA com ViT alcançou uma acurácia média de **90.12%** na tarefa de reconhecimento de doenças de plantas.
- **Eficiência de Dados:** O alto desempenho foi obtido utilizando apenas **cinco imagens de treinamento por doença** (5-shot learning), demonstrando sua eficácia em cenários de dados escassos.
- **Eficiência Computacional:** O tempo de inferência foi de **1.11 ms por imagem** para o ViT, indicando potencial para detecção em tempo real.
- **Citações:** O artigo de 2024 (Rezaei et al.) já possui **81 citações** (em novembro de 2025), indicando um alto impacto na comunidade de pesquisa.

## Features

- **PMF+FA (ViT):** Arquitetura de *Vision Transformer* (ViT) com um pipeline de três estágios (Pré-treinamento, Meta-aprendizado, Ajuste Fino) e um Módulo de Atenção de Características (FA) para focar em áreas discriminativas da imagem.
- **Aprendizado de Meta-aprendizado (Meta-Learning):** Capacidade de aprender a aprender, permitindo que o modelo se adapte rapidamente a novas classes de doenças com poucas amostras.
- **Transferência de Conhecimento:** Utilização de modelos pré-treinados em grandes conjuntos de dados (como ImageNet) para extrair características robustas.
- **Eficácia em Cenários de Campo:** O módulo FA é projetado para mitigar o impacto de fundos complexos e variáveis, comuns em imagens coletadas no campo.

## Use Cases

- **Diagnóstico Precoce de Doenças Raras:** Identificação de novas ou raras doenças de plantas que não possuem um grande histórico de dados rotulados.
- **Monitoramento Agrícola em Tempo Real:** Aplicação em sistemas de agricultura de precisão e robôs de campo para diagnosticar problemas com intervenção mínima do usuário.
- **Adaptação a Novas Culturas/Regiões:** Rápida adaptação de modelos de diagnóstico a novas espécies de plantas ou ambientes geográficos onde os dados de doenças são limitados.
- **Classificação de Insetos e Pragas:** A metodologia FSL é aplicável a outros problemas de classificação na agricultura com escassez de dados, como a identificação de insetos e pragas.

## Integration

A integração de modelos FSL, como o PMF+FA, geralmente segue o pipeline de três estágios:

1.  **Pré-treinamento (Pre-training):** Treinar um extrator de características (por exemplo, ResNet ou ViT) em um grande conjunto de dados de base (por exemplo, PlantVillage) para aprender representações gerais.
2.  **Meta-aprendizado (Meta-learning):** Treinar o modelo (por exemplo, uma *Prototypical Network*) em tarefas de *few-shot* simuladas, usando o conjunto de dados de base. O código de referência para o pipeline PMF (sem o módulo FA) está disponível em: `https://github.com/hushell/pmf_cvpr22`.
3.  **Ajuste Fino (Fine-tuning):** Ajustar o modelo final usando as poucas amostras da nova doença rara.

**Exemplo Conceitual (Python/PyTorch):**

```python
# Exemplo conceitual de como seria a estrutura de um modelo FSL baseado em Prototypical Networks
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_images, query_images, n_way, n_shot, n_query):
        # 1. Extrair características
        all_features = self.feature_extractor(torch.cat([support_images, query_images], dim=0))
        support_features = all_features[:n_way * n_shot]
        query_features = all_features[n_way * n_shot:]

        # 2. Calcular Protótipos (média das características de suporte por classe)
        prototypes = support_features.view(n_way, n_shot, -1).mean(dim=1)

        # 3. Calcular distâncias (por exemplo, distância euclidiana)
        dists = torch.cdist(query_features, prototypes)

        # 4. Converter distâncias em probabilidades (usando softmax sobre o negativo da distância)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

# O Feature Extractor seria um ViT ou ResNet pré-treinado.
# O módulo FA (Feature Attention) seria integrado ao Feature Extractor.
```

## URL

https://www.sciencedirect.com/science/article/pii/S0168169924002035