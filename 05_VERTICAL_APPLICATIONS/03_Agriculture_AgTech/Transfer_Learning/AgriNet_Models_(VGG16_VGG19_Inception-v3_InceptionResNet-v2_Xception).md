# AgriNet Models (VGG16, VGG19, Inception-v3, InceptionResNet-v2, Xception)

## Description

AgriNet é um conjunto de modelos pré-treinados em arquiteturas ImageNet (VGG16, VGG19, Inception-v3, InceptionResNet-v2 e Xception) que foram ajustados e treinados no conjunto de dados AgriNet. O AgriNet dataset é uma coleção massiva de 160.000 imagens agrícolas de mais de 19 localizações geográficas, abrangendo mais de 423 classes de espécies e doenças de plantas, pragas e ervas daninhas. O objetivo principal é fornecer modelos específicos de domínio para superar a limitação de dados e a ausência de modelos pré-treinados específicos para plantas, que são desafios comuns na automação agrícola. O trabalho demonstra a superioridade dos modelos AgriNet em comparação com os modelos base ImageNet em tarefas agrícolas.

## Statistics

- **Precisão de Classificação:** AgriNet-VGG19 alcançou a maior precisão de 94% e o maior F1-score de 92% na classificação das 423 classes.
- **Precisão Mínima:** Todos os modelos AgriNet tiveram uma precisão mínima de 87% (Inception-v3).
- **Dataset:** AgriNet dataset com 160.000 imagens agrícolas e 423 classes.
- **Citações:** O artigo original (arXiv:2207.03881) foi citado 48 vezes (em 2022, o número atual pode ser maior).
- **Publicação:** Aceito pela Frontiers in Plant Science.

## Features

- **Modelos Pré-treinados Específicos de Domínio:** Conjunto de modelos ajustados para o domínio agrícola (AgriNet-VGG16, AgriNet-VGG19, etc.).
- **Dataset Abrangente:** Utiliza o AgriNet dataset com 160k imagens e 423 classes.
- **Transfer Learning:** Aplica o conceito de Transfer Learning para acelerar o treinamento e melhorar a precisão em tarefas de classificação agrícola.
- **Ampla Cobertura de Classes:** Classificação de 423 classes de espécies de plantas, doenças, pragas e ervas daninhas.

## Use Cases

- **Detecção de Doenças de Plantas:** Identificação precisa de várias doenças em diferentes culturas.
- **Classificação de Espécies de Plantas:** Diferenciação de espécies de plantas em ambientes agrícolas.
- **Detecção de Pragas e Ervas Daninhas:** Identificação de pragas e ervas daninhas para manejo de culturas.
- **Automação Agrícola:** Fornecimento de modelos de visão computacional específicos de domínio para sistemas automatizados de monitoramento e diagnóstico.

## Integration

```python
# Exemplo de Integração Simulada para AgriNet (Python/PyTorch-like)
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# 1. Definição do Modelo AgriNet (Simulação de Fine-tuning)
def load_agri_net_model(num_classes=423, model_name='vgg19'):
    if model_name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Substituir a camada final para o número de classes do AgriNet
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    # ... (outras arquiteturas como resnet50)
    
    # Na prática, você carregaria os pesos específicos do AgriNet aqui:
    # model.load_state_dict(torch.load('agri_net_vgg19_weights.pth'))
    
    model.eval()
    return model

# 2. Pré-processamento da Imagem
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Função de Inferência (Simulada)
def predict_plant_disease(image_path, model, class_names):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
    return class_names[predicted_idx.item()]

if __name__ == '__main__':
    simulated_classes = [f"Classe_{i}" for i in range(423)]
    agri_net_model = load_agri_net_model(num_classes=len(simulated_classes), model_name='vgg19')
    print(f"Modelo AgriNet ({agri_net_model.__class__.__name__}) carregado com sucesso.")
    # Para executar a predição, é necessário um arquivo de imagem de planta.
```
**Nota:** Este é um exemplo conceitual. A implementação real requer os pesos do modelo e a lista de classes exata, que devem ser obtidos no repositório oficial do artigo.

## URL

https://arxiv.org/abs/2207.03881