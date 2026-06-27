# ImageNet (Image Classification)

## Description
O ImageNet é um vasto banco de dados visual projetado para uso em pesquisa de software de reconhecimento de objetos visuais. É um esforço de pesquisa contínuo para fornecer dados de imagem para o treinamento de modelos de reconhecimento de objetos em larga escala. A versão mais utilizada é o subconjunto **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** 2012-2017, que se tornou o padrão de fato para benchmarking em visão computacional. O ImageNet é organizado de acordo com a hierarquia do WordNet, onde cada nó (synset) é ilustrado por centenas de milhares de imagens. O impacto do ImageNet foi fundamental para o avanço do Deep Learning, especialmente após a vitória do AlexNet no ILSVRC 2012. Pesquisas recentes (2023-2025) continuam a usar o ImageNet como base, mas também exploram variações como o **ImageNet-D** (CVPR 2024), que utiliza modelos generativos para criar imagens sintéticas e testar a robustez de redes neurais contra distribuições de dados mais desafiadoras.

## Statistics
- **Tamanho Total (Original):** 14.197.122 imagens.
- **Synsets (Categorias):** 21.841 synsets indexados (nós da hierarquia WordNet).
- **Versão Mais Utilizada (ILSVRC 2012/ImageNet-1K):**
    - **Classes:** 1.000 classes de objetos.
    - **Imagens de Treinamento:** 1.281.167 imagens.
    - **Imagens de Validação:** 50.000 imagens.
    - **Imagens de Teste:** 100.000 imagens.
- **Variações Recentes (2024):** ImageNet-D (para robustez), ImageNet-BG (para variações de fundo).

## Features
- **Escala e Diversidade:** Contém milhões de imagens e dezenas de milhares de categorias (synsets).
- **Hierarquia WordNet:** Organizado em uma estrutura hierárquica que permite o treinamento em diferentes níveis de granularidade.
- **ILSVRC-2012 (ImageNet-1K):** Subconjunto mais popular com 1000 classes e mais de 1,2 milhões de imagens de treinamento.
- **Benchmarking de Robustez:** O surgimento de variações como o ImageNet-D (2024) e ImageNet-C/P/A/R/Sketch/Adversarial demonstra o uso contínuo do ImageNet como base para avaliar a robustez e generalização de modelos de Visão Computacional.
- **Imagens Anotadas:** Imagens anotadas manualmente para classificação e localização de objetos.

## Use Cases
- **Treinamento de Modelos de Classificação:** O caso de uso primário, sendo o dataset de referência para treinar e avaliar modelos de classificação de imagens (e.g., ResNet, VGG, EfficientNet).
- **Transfer Learning:** Uso de modelos pré-treinados no ImageNet como *backbones* para tarefas em outros domínios (e.g., detecção de objetos, segmentação semântica, classificação médica) através de *fine-tuning*.
- **Benchmarking de Robustez:** Avaliação da capacidade de modelos de manterem o desempenho sob diferentes tipos de corrupção, distorção ou variações de domínio (e.g., usando ImageNet-C, ImageNet-D).
- **Pesquisa em Visão Computacional:** Desenvolvimento de novas arquiteturas de redes neurais e métodos de treinamento.
- **Modelos de Texto para Imagem (2025):** Pesquisas recentes exploram o uso do ImageNet para treinar modelos de geração de imagem a partir de texto, apesar de seu tamanho relativamente pequeno para essa tarefa.

## Integration
O subconjunto mais comum, o ImageNet ILSVRC 2012 (ImageNet-1K), está disponível para download no **Kaggle** (após aceitar os termos de uso e licença). Para o acesso ao dataset completo e outros subconjuntos, é necessário fazer login ou solicitar acesso no site oficial, concordando com os termos de uso que restringem o uso a **fins de pesquisa e educação não-comerciais**.

**Exemplo de Uso (PyTorch):**
A biblioteca `torchvision` fornece uma classe `torchvision.datasets.ImageNet` para fácil integração, mas o usuário deve primeiro baixar e organizar os dados localmente.

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# O diretório deve conter as pastas 'train' e 'val' com as imagens
traindir = '/path/to/imagenet/train'
valdir = '/path/to/imagenet/val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageNet(
    traindir, split='train', download=False,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
```

## URL
[https://www.image-net.org/](https://www.image-net.org/)
