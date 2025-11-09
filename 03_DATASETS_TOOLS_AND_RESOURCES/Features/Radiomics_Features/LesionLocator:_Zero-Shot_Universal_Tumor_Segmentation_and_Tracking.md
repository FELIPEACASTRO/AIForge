# LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking

## Description

O LesionLocator é um framework inovador de aprendizado profundo para segmentação e rastreamento longitudinal de lesões (tumores) em imagens médicas 3D de corpo inteiro. Ele é notável por sua capacidade de **segmentação universal de tumores zero-shot** (sem treinamento específico para novos tipos de lesão), utilizando *prompts* espaciais densos (como pontos ou caixas) para identificar e rastrear lesões ao longo de exames de acompanhamento (4D). O modelo foi treinado em um extenso conjunto de dados de 23.262 exames médicos anotados, além de dados longitudinais sintéticos, o que lhe confere uma alta generalização. O framework estabelece um novo *benchmark* em segmentação universal e rastreamento longitudinal automatizado de lesões.

## Statistics

Modelo treinado em um extenso dataset de 23.262 exames médicos anotados e dados longitudinais sintéticos. O dataset de treinamento abrange 47 datasets diversos, com diferentes modalidades de imagem e alvos anatômicos. O modelo alcança desempenho de nível humano em segmentação de lesões.

## Features

Segmentação Universal Zero-Shot de Lesões; Rastreamento Longitudinal 4D de Lesões; Utiliza *prompts* espaciais densos (ponto ou caixa); Arquitetura de aprendizado profundo generalizável; Alto desempenho em segmentação (supera modelos existentes em quase 10 pontos Dice); Solução de código aberto (modelo e dataset sintético 4D).

## Use Cases

Segmentação e rastreamento automatizado de tumores em exames de acompanhamento (longitudinais); Auxílio ao diagnóstico e monitoramento de diversas lesões tumorais em imagens médicas 3D de corpo inteiro; Pesquisa e desenvolvimento de novas abordagens de IA em oncologia e radiologia.

## Integration

O código e o *checkpoint* do modelo pré-treinado estão disponíveis no GitHub e Zenodo. O modelo é projetado para ser o primeiro de acesso aberto para segmentação universal de lesões e rastreamento longitudinal automatizado. O uso envolve a aplicação do modelo pré-treinado em novas imagens médicas 3D, fornecendo *prompts* espaciais (ponto ou caixa) para a lesão de interesse. O framework é compatível com o rastreamento de lesões em exames de acompanhamento.

## URL

https://arxiv.org/abs/2502.20985