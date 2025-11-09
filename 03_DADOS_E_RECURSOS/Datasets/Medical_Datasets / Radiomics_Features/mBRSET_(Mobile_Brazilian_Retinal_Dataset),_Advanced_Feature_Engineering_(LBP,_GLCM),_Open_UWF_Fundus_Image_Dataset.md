# mBRSET (Mobile Brazilian Retinal Dataset), Advanced Feature Engineering (LBP, GLCM), Open UWF Fundus Image Dataset

## Description

O primeiro dataset público de retinopatia diabética (RD) capturado usando câmeras portáteis de fundo de olho em cenários de alta demanda no Brasil. O dataset visa preencher a lacuna de dados oftalmológicos em países de baixa e média renda (LMICs). Além disso, o resultado inclui uma técnica de engenharia de features avançada (LBP, GLCM) e um dataset de imagens de campo ultra-amplo (UWF) de 2024.

## Statistics

mBRSET: 5.164 imagens de 1.291 pacientes. 76.79% das imagens sem RD.
Feature Engineering: O modelo mais bem-sucedido (Random Forest com LBP e GLCM) alcançou uma acurácia de 80.41%.
UWF Dataset: 700 imagens UWF de alta resolução (3900 × 3072 pixels) de 602 pacientes. Inclui 7 tipos de pacientes/doenças, incluindo RD.

## Features

Imagens de fundo de olho capturadas com câmeras portáteis. Rótulos de gravidade de RD (ICDR score) e edema macular. Dados clínicos e demográficos detalhados. (mBRSET)
Algoritmos de extração de features: LBP (Local Binary Patterns), GLCM (Gray-Level Co-Occurrence Matrix), PCA (Principal Component Analysis), MAP (Maximum a Posteriori) e GLRLM (Gray-Level Run-Length Matrix). (Feature Engineering)
Imagens UWF (200° do campo retiniano) com informações clínicas e avaliação de qualidade de imagem. (UWF Dataset)

## Use Cases

Desenvolvimento de modelos de IA para triagem e diagnóstico de RD em ambientes de telessaúde e comunitários. Pesquisa sobre disparidades de saúde e fatores de risco. (mBRSET)
Diagnóstico precoce de RD em ambientes clínicos com recursos computacionais limitados. (Feature Engineering)
Desenvolvimento de sistemas de IA para detecção de múltiplas doenças do fundo de olho e treinamento de sistemas automáticos de avaliação de qualidade de imagem (IQA). (UWF Dataset)

## Integration

mBRSET: Acessível via Physionet. Consiste em um arquivo CSV de metadados e um diretório de imagens .jpg.
Feature Engineering: Conceitual. As técnicas LBP e GLCM são implementáveis em bibliotecas de processamento de imagem como OpenCV ou scikit-image em Python.
UWF Dataset: Acessível via Figshare (mencionado no artigo).

## URL

https://www.nature.com/articles/s41597-025-04627-3, https://www.nature.com/articles/s41598-025-06973-z, https://www.nature.com/articles/s41597-024-04113-2