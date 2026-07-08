# VAE Variants - Beta-VAE, VQ-VAE, Hierarchical VAE

## Description

A collection of advanced Variational Autoencoder (VAE) variants designed to overcome the limitations of the original VAE, focusing on more interpretable, discrete, and hierarchical latent representations. **Beta-VAE** (β-VAE) uses a hyperparameter $\\beta$ to enforce *disentanglement* of the latent representations, making them more interpretable. **VQ-VAE** (Vector Quantized VAE) introduces a discrete latent space through vector quantization, solving the 'posterior collapse' problem and being crucial for generating discrete data (such as images and speech). The **Hierarchical VAE** (e.g., NVAE) uses a deep architecture with multiple levels of latent variables to model complex data at different scales, achieving state-of-the-art results in high-resolution image generation.

## Statistics

**Beta-VAE:** $\\beta$ acts as a multiplier for the KL divergence term. Higher values of $\\beta$ result in greater *disentanglement* and more compact latent representations, but can lead to poorer reconstruction quality (a trade-off between *disentanglement* and reconstruction fidelity). **VQ-VAE:** Solves the 'posterior collapse' problem common in VAEs with powerful autoregressive decoders. Discretization allows the model to learn latent representations better suited for data with an inherently discrete structure. **Hierarchical VAE (NVAE):** Achieved state-of-the-art results among non-autoregressive likelihood-based models. On CIFAR-10, it improved the result from 2.98 to 2.91 bits per dimension (BPD) and was the first successful VAE applied to natural images up to 256x256 pixels.

## Features

**Beta-VAE:** Unsupervised *disentangled* representation learning; explicit control over the degree of *disentanglement* through the hyperparameter $\\beta$; more stable to train than approaches such as InfoGAN. **VQ-VAE:** Discrete latent representation; use of a *codebook* (dictionary of embeddings); avoids the *posterior collapse* problem; discretization via *nearest-neighbor* and *straight-through estimator*. **Hierarchical VAE (NVAE):** Deep hierarchical structure for latent variables; use of *depth-wise separable convolutions* and *batch normalization*; residual parameterization of normal distributions; spectral regularization.

## Use Cases

**Beta-VAE:** Automated discovery of basic visual concepts; zero-shot knowledge transfer; image generation controlled by specific latent attributes; learning representations for *reinforcement learning* tasks. **VQ-VAE:** High-quality image generation (a key component in models like DALL-E and VQ-GAN); video and speech generation; unsupervised phoneme learning; lossy data compression. **Hierarchical VAE (NVAE):** High-resolution, high-quality image generation; multi-view document modeling; human motion modeling; learning hierarchical features of complex data.

## Integration

**Beta-VAE:** Integration involves adding the hyperparameter $\\beta$ to the standard VAE loss function, multiplying the KL divergence term. **Code Example (PyTorch - Loss Function Pseudocode):** `loss = reconstruction_loss + beta * kl_divergence`. **VQ-VAE:** The main integration is the vector quantization layer. The loss is composed of three terms: reconstruction loss, VQ loss (*codebook loss*), and *commitment loss*. **Code Example (PyTorch - Quantization Pseudocode):** `quantized_latents = z_e + (z_q - z_e).detach()`. **Hierarchical VAE (NVAE):** Requires building a deep network with multiple levels of latent variables and careful architectural design with residual blocks and normalization. The official NVAE code is available on GitHub (NVlabs/NVAE).

## URL

Beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl | VQ-VAE: https://arxiv.org/abs/1711.00937 | NVAE: https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html
