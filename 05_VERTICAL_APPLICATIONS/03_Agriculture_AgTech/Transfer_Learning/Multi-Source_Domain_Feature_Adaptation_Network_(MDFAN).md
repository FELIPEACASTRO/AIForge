# Multi-Source Domain Feature Adaptation Network (MDFAN)

## Description

A Rede de Adaptação de Características de Domínio de Múltiplas Fontes (MDFAN) é uma arquitetura de aprendizado profundo proposta para resolver o problema da baixa precisão no reconhecimento de doenças em imagens agrícolas de campo, causada pela mudança de domínio (domain shift) entre os dados de treinamento e os dados de aplicação real. O MDFAN emprega uma estratégia de alinhamento em duas etapas: primeiro, alinha a distribuição de cada par de domínio-fonte-alvo em múltiplos espaços de características específicos, utilizando extração de múltiplas representações e alinhamento de subdomínios; segundo, alinha as saídas do classificador aproveitando os limites de decisão dentro de domínios específicos. Esta abordagem é robusta a variações de condições de iluminação e permite a adaptação não supervisionada de múltiplas fontes (MUDA).

## Statistics

Acurácia Média de Classificação: **92,11%** com dois domínios-fonte e **93,02%** com três domínios-fonte. O desempenho superou todos os outros métodos testados no estudo. O artigo foi publicado em 2024 e possui 1 citação (até a última verificação).

## Features

Adaptação de Domínio Não Supervisionada de Múltiplas Fontes (MUDA); Estratégia de alinhamento em duas etapas (características e saídas do classificador); Robustez a mudanças nas condições de iluminação; Extração de múltiplas representações; Alinhamento de subdomínios.

## Use Cases

Reconhecimento de doenças da batata em ambientes de campo, especificamente para cinco tipos distintos de doenças. Aplica-se a cenários onde a transferência de conhecimento entre diferentes regiões, estações ou condições de iluminação é necessária.

## Integration

O MDFAN é uma rede de aprendizado profundo e, embora o código de implementação específico não tenha sido encontrado em repositórios públicos como o GitHub, a técnica é baseada em arquiteturas de redes neurais convolucionais (CNNs) e pode ser implementada usando frameworks populares como PyTorch ou TensorFlow. A integração envolveria a adaptação do código-fonte para a estratégia de alinhamento de duas etapas e a aplicação em um novo conjunto de dados de campo (domínio-alvo) para a tarefa de reconhecimento de doenças.

**Exemplo de Estrutura de Código (Conceitual em PyTorch):**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definir o Extrator de Características (Feature Extractor)
class FeatureExtractor(nn.Module):
    # ... (Implementação baseada em ResNet ou EfficientNet)
    pass

# 2. Definir o Classificador (Classifier)
class Classifier(nn.Module):
    # ... (Camadas densas para classificação)
    pass

# 3. Definir o Módulo de Alinhamento de Domínio (Domain Alignment Module)
class DomainAlignment(nn.Module):
    # ... (Implementação do alinhamento de duas etapas do MDFAN)
    pass

# 4. Função de Treinamento (Conceptual)
def train_mdfan(source_data_list, target_data_unlabeled):
    # Inicialização de modelos e otimizadores
    extractor = FeatureExtractor()
    classifier = Classifier()
    aligner = DomainAlignment()
    
    # Otimizador e função de perda
    optimizer = optim.Adam(list(extractor.parameters()) + list(classifier.parameters()) + list(aligner.parameters()))
    
    for epoch in range(num_epochs):
        # 1. Passo de Alinhamento de Características (Feature Alignment)
        # Calcular perdas de alinhamento de subdomínios
        
        # 2. Passo de Alinhamento de Saída do Classificador (Classifier Output Alignment)
        # Calcular perdas de discrepância de preditor
        
        # 3. Passo de Classificação (Classification)
        # Calcular perda de classificação no domínio-fonte
        
        # Otimização
        optimizer.zero_grad()
        # Perda total = Perda de Classificação + Perda de Alinhamento
        # total_loss.backward()
        optimizer.step()
```

## URL

https://doi.org/10.3389/fpls.2024.1471085