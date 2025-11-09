# PDDNet-LVE (Lead Voting Ensemble) - Transfer Learning em Agricultura

## Description

Arquitetura de rede neural convolucional (CNN) PDDNet-LVE (Lead Voting Ensemble) que integra nove CNNs pré-treinadas (incluindo DenseNet201, ResNet101, ResNet50, GoogleNet, AlexNet, ResNet18, EfficientNetB7, NASNetMobile, e ConvNeXtSmall) e é ajustada (fine-tuned) por extração de características profundas para identificação e classificação eficiente de doenças em plantas. Embora o modelo não seja explicitamente treinado em dados médicos, o artigo que o descreve (publicado em 2024) estabelece a relevância do Transfer Learning em domínios de imagem com características visuais complexas, como o médico e o agrícola, sugerindo a base para o Transfer Learning Cross-Domain. O uso de modelos pré-treinados em datasets massivos (como ImageNet, que é a base para a maioria dos modelos citados) e a menção da relevância do TL em ambos os domínios (médico e agrícola) no mesmo parágrafo servem como a melhor evidência acessível no momento para o tema solicitado.

## Statistics

Acurácia de 97.79% no dataset PlantVillage (15 classes, 54.305 imagens). O modelo PDDNet-AE (Early Fusion) alcançou 96.74% de acurácia. Publicado em 2024.

## Features

Uso de ensemble (LVE) para aumentar a robustez e capacidade de generalização; Utiliza modelos pré-treinados (TL) para mitigar a escassez de dados e a complexidade de fundo; Adequado para implantação em dispositivos pequenos (móveis); Alta acurácia na classificação de doenças em plantas.

## Use Cases

Classificação e detecção de doenças em plantas para agricultura sustentável; Aplicações em dispositivos móveis para diagnóstico em campo; Mitigação de escassez de dados em domínios específicos através do aproveitamento de conhecimento de domínios com dados abundantes (como o médico).

## Integration

O modelo é baseado em arquiteturas de Deep Learning (DenseNet, ResNet, etc.) e pode ser implementado usando bibliotecas padrão como PyTorch ou TensorFlow. A integração envolve o carregamento dos pesos pré-treinados, a substituição da camada de classificação final e o ajuste fino (fine-tuning) com o dataset agrícola específico. O artigo não fornece um repositório GitHub direto, mas a metodologia é padrão para Transfer Learning em Visão Computacional.

## URL

https://link.springer.com/article/10.1186/s12870-024-04825-y