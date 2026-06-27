# VAE Variants - Beta-VAE, VQ-VAE, Hierarchical VAE

## Description

Uma coleção de variantes avançadas de Autoencoders Variacionais (VAEs) projetadas para superar as limitações do VAE original, focando em representações latentes mais interpretáveis, discretas e hierárquicas. O **Beta-VAE** (β-VAE) utiliza um hiperparâmetro $\\beta$ para forçar o *disentanglement* (desemaranhamento) das representações latentes, tornando-as mais interpretáveis. O **VQ-VAE** (Vector Quantized VAE) introduz um espaço latente discreto através da quantização vetorial, resolvendo o problema de 'colapso posterior' e sendo crucial para a geração de dados discretos (como imagens e fala). O **VAE Hierárquico** (ex: NVAE) usa uma arquitetura profunda com múltiplos níveis de variáveis latentes para modelar dados complexos em diferentes escalas, alcançando resultados de última geração em geração de imagens de alta resolução.

## Statistics

**Beta-VAE:** O $\\beta$ atua como um multiplicador para o termo de divergência KL. Valores mais altos de $\\beta$ resultam em maior *disentanglement* e representações latentes mais compactas, mas podem levar a uma pior qualidade de reconstrução (trade-off entre *disentanglement* e fidelidade de reconstrução). **VQ-VAE:** Resolve o problema de 'colapso posterior' comum em VAEs com decodificadores autorregressivos poderosos. A discretização permite que o modelo aprenda representações latentes mais adequadas para dados com estrutura inerentemente discreta. **VAE Hierárquico (NVAE):** Alcançou resultados de última geração (*state-of-the-art*) entre modelos baseados em verossimilhança não autorregressivos. No CIFAR-10, melhorou o resultado de 2.98 para 2.91 bits por dimensão (BPD) e foi o primeiro VAE bem-sucedido aplicado a imagens naturais de até 256x256 pixels.

## Features

**Beta-VAE:** Aprendizagem de representação *disentangled* não supervisionada; controle explícito sobre o grau de *disentanglement* através do hiperparâmetro $\\beta$; mais estável de treinar do que abordagens como InfoGAN. **VQ-VAE:** Representação latente discreta; uso de um *codebook* (dicionário de embeddings); evita o problema de *posterior collapse*; discretização via *nearest-neighbor* e *straight-through estimator*. **VAE Hierárquico (NVAE):** Estrutura hierárquica profunda para variáveis latentes; uso de convoluções separáveis em profundidade (*depth-wise separable convolutions*) e *batch normalization*; parametrização residual de distribuições normais; regularização espectral.

## Use Cases

**Beta-VAE:** Descoberta automatizada de conceitos visuais básicos; transferência de conhecimento zero-shot; geração de imagens controlada por atributos latentes específicos; aprendizado de representações para tarefas de *reinforcement learning*. **VQ-VAE:** Geração de imagens de alta qualidade (componente chave em modelos como DALL-E e VQ-GAN); geração de vídeo e fala; aprendizado não supervisionado de fonemas; compressão de dados com perdas. **VAE Hierárquico (NVAE):** Geração de imagens de alta resolução e qualidade; modelagem de documentos multi-view; modelagem de movimento humano; aprendizado de características hierárquicas de dados complexos.

## Integration

**Beta-VAE:** A integração envolve adicionar o hiperparâmetro $\\beta$ à função de perda VAE padrão, multiplicando o termo de divergência KL. **Exemplo de Código (PyTorch - Pseudocódigo da Função de Perda):** `loss = reconstruction_loss + beta * kl_divergence`. **VQ-VAE:** A integração principal é a camada de quantização vetorial. A perda é composta por três termos: perda de reconstrução, perda de VQ (*codebook loss*) e perda de comprometimento (*commitment loss*). **Exemplo de Código (PyTorch - Pseudocódigo da Quantização):** `quantized_latents = z_e + (z_q - z_e).detach()`. **VAE Hierárquico (NVAE):** Requer a construção de uma rede profunda com múltiplos níveis de variáveis latentes e um design arquitetônico cuidadoso com blocos residuais e normalização. O código oficial do NVAE está disponível no GitHub (NVlabs/NVAE).

## URL

Beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl | VQ-VAE: https://arxiv.org/abs/1711.00937 | NVAE: https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html