# Mob-Res

## Description

Modelo de Rede Neural Convolucional (CNN) leve e explicável, que combina aprendizado residual (residual blocks) com o extrator de características MobileNetV2. Projetado para diagnóstico de doenças em plantas, sendo adequado para aplicações móveis devido ao seu baixo número de parâmetros.

## Statistics

Parâmetros: 3.51 milhões. Acurácia Média: 97.73% (Plant Disease Expert, 58 classes). Acurácia: 99.47% (PlantVillage, 38 classes). Publicação: Scientific Reports (2025).

## Features

Arquitetura Híbrida (blocos residuais e MobileNetV2); Leveza (3.51 milhões de parâmetros); Explicabilidade (Explainable AI).

## Use Cases

Diagnóstico rápido e preciso de doenças em plantas no campo; Aplicações móveis para assistência a agricultores; Prevenção de surtos e proteção de colheitas.

## Integration

Requisitos: Python 3.10, TensorFlow 2.10, Keras 2.10, Numpy 1.26. Código e modelo pré-treinado (.keras) disponíveis no repositório GitHub.

## URL

https://www.nature.com/articles/s41598-025-94083-1