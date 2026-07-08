# Normalizing Flows (RealNVP, Glow, CNF)

## Description

**Normalizing Flows (NFs)** are a class of generative models that build complex probability distributions from a simple base distribution (usually Gaussian) through a sequence of invertible and differentiable transformations [1] [2]. Their main value proposition is enabling **exact and efficient probability density (likelihood) evaluation** and **exact sampling/inference** of latent variables, something that models like VAEs and GANs do not offer [3]. The latent space is well-behaved and has the same dimensionality as the data space. **RealNVP** [5] was one of the first to use affine coupling layers to guarantee invertibility and easy computation of the Jacobian. **Glow** [4] improved on RealNVP with invertible 1x1 convolutions and Actnorm to enhance the quality of generated images. **Continuous Normalizing Flows (CNFs)** [6] model the flow as a continuous-time trajectory using Neural ODEs, offering unlimited depth flexibility with parameter efficiency.

## Statistics

**Training Objective:** Maximization of the Exact Log-Likelihood (NLL) for all models. **Parallelization:** Parallelizable Sampling and Inference (RealNVP, Glow); Sequential Sampling, Parallelizable Inference (CNFs). **Performance (Images):** Glow achieved the best Log-Likelihood on ImageNet and CelebA among early flow models. **Computational Cost:** Variable, with the CNF depending on the ODE Solver's precision, but generally more parameter-efficient than discrete models. **Invertibility:** Guaranteed by construction in all models.

## Features

**Normalizing Flows (General):** Exact and tractable likelihood; Exact and parallelized sampling and inference; Meaningful and manipulable latent space. **RealNVP:** Use of affine coupling layers; Multi-scale architecture for high-dimensional data. **Glow:** Introduction of invertible 1x1 convolutions to permute channels; Use of Actnorm (Activation Normalization); Improved image generation quality. **CNFs:** Use of Neural ODEs to define continuous-time transformations; Unlimited depth (time) flexibility without increased training cost; Efficient use of parameters.

## Use Cases

**High-Fidelity Image Generation:** Glow is notable for generating realistic faces and enabling semantic manipulations in the latent space (e.g., changing facial expression) [4]. **Voice and Audio Synthesis:** Used in models such as WaveGlow [7]. **Anomaly and Outlier Detection:** The ability to compute the exact likelihood $p(x)$ is ideal for identifying low-probability data points [1]. **Variational Inference:** CNFs can be used for more expressive and accurate variational inference [6]. **Modeling Physical and Chemical Distributions:** Applications in computational physics to model energy distributions and enable efficient sampling in Monte Carlo simulations [8]. **Lossless Data Compression:** Explicit density modeling allows use for data compression, where the log-likelihood relates directly to the bit rate [1].

## Integration

Integration is typically done using deep learning libraries such as PyTorch or TensorFlow. Ready-to-use implementations are available in libraries like `nflows` [11]. Training is based on maximizing the exact log-likelihood. CNFs require Ordinary Differential Equation (ODE) solver libraries, such as `torchdiffeq`, to solve the integral of the Jacobian trace.

**Conceptual Example (PyTorch - Glow-like Flow):**
```python
import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, AffineCouplingTransform, ActNorm, OneByOneConvolution

# 1. Define the Base Distribution
base_dist = StandardNormal(shape=[2])

# 2. Define the Transformations (Example of one Glow step)
transform = CompositeTransform([
    ActNorm(2),
    OneByOneConvolution(2),
    AffineCouplingTransform(...) # Affine coupling layer
])

# 3. Build the Flow
flow = Flow(transform, base_dist)

# 4. Training (Maximize the Log-Likelihood)
# loss = -flow.log_prob(data).mean()
```

## URL

https://arxiv.org/abs/1908.09257
