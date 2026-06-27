# DDPM (Denoising Diffusion Probabilistic Models)

## Description

Modelo generativo fundamental baseado em termodinâmica de não-equilíbrio. Define um processo de difusão (forward process) que adiciona ruído Gaussiano em uma cadeia de Markov e um processo reverso que aprende a remover o ruído para gerar dados. Sua proposta de valor única é a estabilidade e a qualidade de geração de imagens, superando as GANs em certas métricas de qualidade.

## Statistics

Métrica de Qualidade: FID (Fréchet Inception Distance) baixo, indicando alta qualidade de imagem. Tempo de Amostragem: Lento, tipicamente requer 1000 passos de amostragem para resultados de alta qualidade.

## Features

Geração de imagens de alta qualidade, treinamento estável, capacidade de modelar distribuições de dados complexas. Processo de amostragem estocástico (Markoviano).

## Use Cases

Geração de imagens incondicional e condicional (texto-para-imagem), restauração de imagens (inpainting, super-resolução), síntese de áudio e vídeo.

## Integration

Implementação fundamental em bibliotecas como PyTorch e TensorFlow. O treinamento envolve a otimização de uma rede neural (geralmente U-Net) para prever o ruído adicionado em cada passo. Exemplo (Conceitual PyTorch): `model = UNet(in_channels=3, out_channels=3, time_emb_dim=256)` e `loss = F.mse_loss(predicted_noise, true_noise)`.

## URL

https://arxiv.org/abs/2006.11239