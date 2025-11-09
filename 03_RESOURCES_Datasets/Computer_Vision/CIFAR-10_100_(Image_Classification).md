# CIFAR-10/100 (Image Classification)

## Description
O **CIFAR-10** e o **CIFAR-100** são datasets de visão computacional amplamente utilizados para o treinamento de modelos de classificação de imagens. Ambos são subconjuntos rotulados do dataset de 80 milhões de pequenas imagens e foram criados por Alex Krizhevsky, Vinod Nair e Geoffrey Hinton [1].

O **CIFAR-10** consiste em 60.000 imagens coloridas (32x32 pixels) em 10 classes mutuamente exclusivas, com 6.000 imagens por classe. As classes incluem: avião, automóvel, pássaro, gato, veado, cachorro, sapo, cavalo, navio e caminhão.

O **CIFAR-100** é semelhante, mas possui 100 classes, cada uma com 600 imagens. As 100 classes são agrupadas em 20 superclasses, fornecendo um rótulo "fino" (a classe específica) e um rótulo "grosso" (a superclasse) para cada imagem. Isso o torna ideal para tarefas de classificação hierárquica [1].

Apesar de sua criação ser anterior a 2023, o dataset continua sendo uma referência fundamental e é constantemente utilizado em pesquisas recentes (2023-2025) para benchmarking de novas arquiteturas de redes neurais, como ResNet e Wide Residual Networks, e para estudos sobre aprendizado com rótulos ruidosos e transferência de aprendizado [2] [3] [4].

## Statistics
- **Tamanho Total:** 60.000 imagens coloridas (32x32 pixels).
- **Divisão:** 50.000 imagens de treino e 10.000 imagens de teste.
- **Resolução:** 32x32 pixels (RGB).
- **CIFAR-10:** 10 classes, 6.000 imagens por classe.
- **CIFAR-100:** 100 classes (rótulos finos) e 20 superclasses (rótulos grossos), 600 imagens por classe fina.
- **Tamanho do Arquivo (Python Version):** CIFAR-10 (163 MB), CIFAR-100 (161 MB).
- **Versões Notáveis:** CIFAR-10.1 (novo conjunto de testes para CIFAR-10) [5].

## Features
- **Imagens Coloridas:** Todas as imagens são coloridas (RGB).
- **Resolução Baixa:** Resolução fixa de 32x32 pixels, o que o torna ideal para testes rápidos e desenvolvimento de prova de conceito.
- **Estrutura de Classes:**
    - **CIFAR-10:** 10 classes de objetos.
    - **CIFAR-100:** 100 classes de objetos, agrupadas em 20 superclasses (rótulos finos e grossos).
- **Divisão Padrão:** 50.000 imagens de treino e 10.000 imagens de teste.
- **Versões Estendidas:** Existem versões estendidas como o CIFAR-10.1, que oferece um novo conjunto de testes para uma avaliação mais robusta dos modelos [5].

## Use Cases
O CIFAR-10/100 é um dataset de referência (benchmark) para o desenvolvimento e avaliação de algoritmos de visão computacional, especialmente em:

- **Classificação de Imagens:** É o caso de uso primário, testando a capacidade de um modelo de atribuir corretamente uma imagem a uma das classes definidas.
- **Desenvolvimento de Arquiteturas CNN:** Utilizado para testar a eficácia de novas arquiteturas de Redes Neurais Convolucionais (CNNs) e modelos de Deep Learning, como ResNet, VGG e Wide Residual Networks [3].
- **Aprendizado por Transferência (Transfer Learning):** Embora pequeno, é frequentemente usado como um dataset de destino para modelos pré-treinados em datasets maiores, como o ImageNet, para avaliar a capacidade de transferência de conhecimento [8].
- **Aprendizado com Rótulos Ruidosos:** Versões modificadas, como o CIFAR-10/100N, são usadas para pesquisar e desenvolver métodos robustos de aprendizado de máquina que podem lidar com erros de anotação [2].
- **Geração de Imagens:** Utilizado para treinar e avaliar modelos generativos, como GANs (Generative Adversarial Networks) e VAEs (Variational Autoencoders), devido à sua estrutura de classes bem definida.
- **Quantização e Otimização de Modelos:** Usado para testar a eficiência de técnicas de compressão e otimização de modelos para implantação em dispositivos com recursos limitados.

## Integration
A maneira mais comum e recomendada de integrar o CIFAR-10/100 é através de bibliotecas de aprendizado de máquina de alto nível, como PyTorch e TensorFlow, que fornecem funções utilitárias para download e carregamento automático do dataset.

**Exemplo de Integração (PyTorch):**
O `torchvision.datasets` permite o download e carregamento do dataset com apenas algumas linhas de código, eliminando a necessidade de gerenciar arquivos manualmente [6].

```python
import torchvision
import torchvision.transforms as transforms

# Define as transformações a serem aplicadas nas imagens
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download e carregamento do dataset de treino (CIFAR-10)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Download e carregamento do dataset de teste (CIFAR-10)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
```

**Exemplo de Integração (TensorFlow/Keras):**
O Keras também oferece acesso direto ao dataset [7].

```python
from tensorflow.keras.datasets import cifar10

# Carrega o dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

Para download manual, os arquivos estão disponíveis no formato Python `pickle`, Matlab e binário no site oficial [1].

## URL
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
