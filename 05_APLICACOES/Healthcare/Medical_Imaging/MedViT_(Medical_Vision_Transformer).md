# MedViT (Medical Vision Transformer)

## Description

O MedViT (Medical Vision Transformer) é um modelo híbrido robusto que combina a capacidade de extração de características locais das Redes Neurais Convolucionais (CNNs) com a conectividade global dos Vision Transformers (ViTs). Ele foi projetado especificamente para a classificação generalizada de imagens médicas, abordando a preocupação com a fragilidade dos modelos de diagnóstico profundo contra ataques adversários. O modelo utiliza um mecanismo de atenção eficiente baseado em convolução para mitigar a alta complexidade quadrática do mecanismo de autoatenção padrão do ViT. Além disso, ele incorpora uma técnica de aumento de forma (shape augmentation) para aprender limites de decisão mais suaves, o que aumenta sua robustez e capacidade de generalização em diversos conjuntos de dados médicos. Uma versão aprimorada, o MedViT-V2, integra as Redes de Kolmogorov-Arnold (KAN) para melhorias na arquitetura e um novo benchmark de corrupção.

## Statistics

- **Publicação:** *Computers in Biology and Medicine*, 2023.
- **Citações:** 220 estrelas no GitHub (em 2025).
- **Desempenho (Pré-treinamento ImageNet-1K):**
    - MedViT\_small: Acc@1 de 83.70%
    - MedViT\_base: Acc@1 de 83.92%
    - MedViT\_large: Acc@1 de 83.96%
- **Vantagem:** Demonstra alta robustez e generalização em comparação com ResNets de linha de base em termos de *trade-off* entre Acurácia/AUC e número de Parâmetros em todos os conjuntos de dados MedMNIST-2D.
- **Complexidade:** Menor complexidade computacional em comparação com ViTs padrão devido ao mecanismo de atenção baseado em convolução.

## Features

- **Modelo Híbrido CNN-Transformer:** Combina a localidade das CNNs com a conectividade global dos ViTs.
- **Mecanismo de Atenção Eficiente:** Utiliza uma operação de convolução eficiente para o mecanismo de autoatenção, reduzindo a complexidade quadrática.
- **Robustez a Ataques Adversários:** Projetado para aprender limites de decisão mais suaves através de aumento de forma (shape augmentation), conferindo alta robustez.
- **Generalização em Imagens Médicas:** Demonstra alta capacidade de generalização em uma ampla coleção de conjuntos de dados MedMNIST-2D.
- **MedViT-V2:** Versão aprimorada que integra as Redes de Kolmogorov-Arnold (KAN) para melhorias arquitetônicas.

## Use Cases

- **Classificação Generalizada de Imagens Médicas:** Diagnóstico automatizado de doenças em diversos conjuntos de dados 2D (por exemplo, MedMNIST-2D).
- **Análise de Imagens com Robustez:** Aplicações onde a confiabilidade contra ataques adversários é crítica, como em sistemas de diagnóstico clínico.
- **Visualização e Interpretabilidade:** Utilização de técnicas como Grad-CAM para inspeção visual e interpretabilidade do diagnóstico em conjuntos de dados médicos.
- **Desenvolvimento de Arquiteturas Híbridas:** Serve como base para o desenvolvimento de modelos que combinam o melhor das CNNs e dos Transformers.

## Integration

O modelo é implementado em PyTorch. O repositório oficial fornece o código-fonte e um *notebook* de instruções (`Instructions.ipynb`) para treinamento e avaliação.

**Exemplo de Uso (Estrutura de Alto Nível):**

```python
import torch
from MedViT import MedViT_small # Assumindo que a classe MedViT_small está no arquivo MedViT.py

# 1. Carregar o modelo
# O modelo MedViT_small pré-treinado no ImageNet-1K
model = MedViT_small(pretrained=True) 
model.eval()

# 2. Preparar a imagem de entrada (exemplo)
# A imagem deve ser pré-processada (redimensionada para 224x224, normalizada)
# Exemplo de tensor de entrada (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224) 

# 3. Realizar a inferência
with torch.no_grad():
    output = model(dummy_input)

# 4. Processar a saída (depende da tarefa de classificação)
# print(output.shape) 
# print(torch.argmax(output, dim=1))

# O repositório também oferece um guia para datasets customizados.
# Para treinamento, seguir as instruções no arquivo 'Instructions.ipynb' ou 'CustomDataset.md'
```

## URL

https://github.com/Omid-Nejati/MedViT