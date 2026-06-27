# nnU-Net (no-new-U-Net)

## Description

Um framework de aprendizado profundo (Deep Learning) auto-configurável e adaptável para segmentação de imagens biomédicas. Ele automatiza o pré-processamento, a arquitetura de rede (baseada em U-Net 2D e 3D), o treinamento e o pós-processamento, eliminando a necessidade de ajustes manuais complexos para cada novo conjunto de dados. É reconhecido por seu desempenho de ponta e robustez em diversos desafios de segmentação de imagens médicas.

## Statistics

Vencedor de 9 de 10 desafios no MICCAI 2020 e 5 de 7 no MICCAI 2021. Vencedor do AMOS2022. Serve como *baseline* robusta para novas arquiteturas de segmentação. Mais de 7.600 estrelas no GitHub. Artigo original (2021) citado mais de 7.600 vezes.

## Features

Auto-configuração de pipeline; Adaptação automática a diferentes conjuntos de dados; Implementação de U-Net 2D e 3D; Suporte a diversas modalidades de imagem (TC, RM, Microscopia); Alto desempenho em competições de segmentação.

## Use Cases

Segmentação de tumores em TC e RM; Segmentação de órgãos em imagens médicas 3D; Análise de imagens de microscopia; Segmentação de nódulos pulmonares. É aplicável a qualquer tarefa de segmentação de imagem biomédica.

## Integration

O nnU-Net é instalado via pip e usado através de comandos de linha de terminal para preparar dados, treinar modelos e realizar inferência. Exemplo de instalação: `pip install nnunetv2`. O uso envolve a definição de variáveis de ambiente e a execução de comandos como `nnunetv2_plan_and_preprocess`, `nnunetv2_train`, e `nnunetv2_predict`. O código-fonte está disponível no GitHub para integração mais profunda.

## URL

https://github.com/MIC-DKFZ/nnUNet