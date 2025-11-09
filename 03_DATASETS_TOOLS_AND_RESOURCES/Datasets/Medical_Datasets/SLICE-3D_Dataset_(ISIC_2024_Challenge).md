# SLICE-3D Dataset (ISIC 2024 Challenge)

## Description

O SLICE-3D Dataset é o conjunto de dados oficial de treinamento para o Desafio ISIC 2024 (International Skin Imaging Collaboration). Ele consiste em recortes de lesões de pele extraídos de fotografias corporais totais 3D (3D TBP), projetado para imitar imagens de smartphones enviadas para telemedicina. O dataset visa o desenvolvimento de algoritmos de aprendizado de máquina para a detecção de câncer de pele, focando em lesões padronizadas de 15mm por 15mm.

## Statistics

Total de 401.059 imagens JPEG de lesões de pele (versão CC-BY-NC). As imagens são recortes de 15mmx15mm. Metadados clínicos incluem idade, sexo, local anatômico geral, identificador de paciente, tamanho clínico e valores de malignidade (gold standard).

## Features

O foco principal é em **Deep Learning (DL)** e **Radiomics**. As técnicas de feature engineering incluem: **Extração de características de imagem** (via CNNs e modelos de atenção), **Características Radiômicas** (como entropia e contraste wavelet), e **Características Híbridas** (combinação de dados de imagem e metadados clínicos tabulares como idade e sexo). Modelos avançados utilizam frameworks de **segmentação e classificação em duas etapas**.

## Use Cases

Desenvolvimento de modelos de **classificação** (maligno/benigno) e **detecção precoce de câncer de pele** em imagens não dermatoscópicas. Aplicações em **telemedicina** e triagem automatizada de lesões de pele a partir de fotos de celular.

## Integration

O dataset está disponível para download em formato de arquivos compactados (1.2GB para imagens, 40MB para metadados suplementares e 7MB para o gabarito de malignidade). O acesso primário e a participação no desafio ocorreram via **Kaggle**. A citação obrigatória para a versão CC-BY-NC é: International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. *International Skin Imaging Collaboration* [https://doi.org/10.34970/2024-slice-3d] (2024).

## URL

https://challenge2024.isic-archive.com/