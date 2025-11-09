# RadImageNet (Modelos Pré-treinados)

## Description

**RadImageNet** é um conjunto de modelos de redes neurais convolucionais (CNNs) pré-treinados exclusivamente em um grande conjunto de dados de imagens médicas (radiológicas). O objetivo é fornecer um ponto de partida mais eficaz para o Transfer Learning (TL) em aplicações de Imagem Médica, superando as limitações dos modelos pré-treinados em ImageNet (imagens naturais). O conjunto de dados RadImageNet original contém **1.35 milhão de imagens anotadas** de Tomografia Computadorizada (TC), Ressonância Magnética (RM) e Ultrassom, cobrindo 3 modalidades, 11 anatomias e 165 patologias. Estudos recentes (2023-2025) indicam que os modelos RadImageNet geralmente demonstram desempenho superior ou comparável ao ImageNet em tarefas radiológicas, especialmente em cenários com dados limitados, e oferecem melhor interpretabilidade.

## Statistics

**Tamanho do Dataset:** 1.35 milhão de imagens médicas (TC, RM, Ultrassom). **Modelos Pré-treinados:** ResNet50, DenseNet121, InceptionResNetV2, InceptionV3. **Desempenho (Top1/Top5 Accuracy no dataset RadImageNet):** InceptionResNetV2: 74.0% / 94.3%; ResNet50: 72.3% / 94.1%; DenseNet121: 73.1% / 96.1%; InceptionV3: 73.2% / 92.7%. **Citações:** O artigo principal (DOI: 10.1148/ryai.210315) foi citado mais de 388 vezes (dado de 2022). **Comparação:** Modelos RadImageNet superam ou igualam o desempenho de modelos ImageNet em tarefas radiológicas, com vantagens em cenários de dados limitados.

## Features

**Pré-treinamento Específico para Domínio:** Treinado exclusivamente em imagens radiológicas, o que permite que os modelos aprendam características específicas do domínio médico. **Diversidade de Modalidades:** Inclui imagens de TC, RM e Ultrassom. **Arquiteturas Populares:** Modelos pré-treinados disponíveis para arquiteturas amplamente utilizadas como ResNet50, DenseNet121, InceptionResNetV2 e InceptionV3. **Melhor Interpretabilidade:** Demonstra melhor interpretabilidade em comparação com modelos ImageNet.

## Use Cases

**Classificação de Lesões:** Classificação de lesões mamárias em ultrassom. **Detecção de Patologias:** Detecção de ruptura do ligamento cruzado anterior (LCA) e menisco em RM. **Diagnóstico Pulmonar:** Detecção de pneumonia em radiografias de tórax e identificação de SARS-CoV-2 em TC de tórax. **Detecção de Hemorragia:** Detecção de hemorragia em TC de cabeça. **Previsão de Malignidade:** Previsão de malignidade de nódulos tireoidianos em ultrassom.

## Integration

**Disponibilidade dos Modelos:** Os modelos pré-treinados para TensorFlow e PyTorch estão disponíveis via Google Drive (links no repositório GitHub oficial). **Exemplo de Código (PyTorch):** O repositório GitHub oficial (BMEII-AI/RadImageNet) contém um notebook de exemplo (`pytorch_example.ipynb`) e scripts de treinamento (`*_train.py`) para diversas aplicações médicas, demonstrando o processo de fine-tuning. **Uso (Conceitual):** Carregar o modelo pré-treinado RadImageNet e realizar o fine-tuning (ajuste fino) em um conjunto de dados específico para a tarefa médica desejada, ajustando a taxa de aprendizado e o número de camadas congeladas.

## URL

https://github.com/BMEII-AI/RadImageNet