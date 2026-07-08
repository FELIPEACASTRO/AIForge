# Energy-Based Models (EBMs) - Contrastive Divergence (CD) and Score Matching (SM)

## Description

Energy-Based Models (EBMs) are a class of unnormalized probabilistic models that define a probability distribution $p_\\theta(x) = \\frac{1}{Z(\\theta)} e^{-E_\\theta(x)}$, where $E_\\theta(x)$ is the energy function (typically a neural network) and $Z(\\theta)$ is the intractable partition function. The unique value proposition of EBMs lies in their flexibility to model complex multimodal distributions, assigning low energy to real data and high energy to unrealistic data, without the architectural constraints of GANs or VAEs. **Contrastive Divergence (CD)** and **Score Matching (SM)** are the two main training methods for EBMs, each addressing the challenge of the intractable partition function in distinct ways.

## Statistics

CD minimizes the KL divergence ($D_{KL}$) between the data distribution and the model distribution, and is known for stability issues (short-chain divergence, CD-k). SM minimizes the Fisher divergence ($D_F$), offering a tractable and monitorable objective function, and serving as the foundation for modern Score-Based Generative Models (such as diffusion models). Variants such as Denoising Score Matching (DSM) and Sliced Score Matching (SSM) improved training stability and efficiency compared to traditional CD.

## Features

EBMs: Implicit probability density modeling; Flexibility in the energy function's architecture; Inherent anomaly detection capability (high-energy data). CD: Maximum likelihood training; Simple to implement (CD-k); Requires sampling (e.g., Langevin Dynamics) to estimate the gradient. SM: Sampling-free training; Tractable and monitorable training objective; Foundation for diffusion models.

## Use Cases

Content Generation: High-quality images, text, and audio. Anomaly Detection (Out-of-Distribution Detection): The energy value can be used as an anomaly metric. Applications in Energy Systems: Probabilistic forecasting of power generation (wind, solar). Motion Optimization: Integration as cost factors or initial sampling distributions in motion planning and robotics problems.

## Integration

Integrating EBMs is typically done in deep learning frameworks such as PyTorch or TensorFlow. Training with CD involves simulating Markov chains (e.g., Langevin Dynamics) to obtain negative samples. Training with SM (especially DSM) is more straightforward, focusing on making the gradient of the energy function (the score) match the score of the noisy data distribution. \n\n**Conceptual Score Matching (DSM) Training Example:**\n```python\nimport torch\n\n# E_theta is the energy function (neural network)\n# sigma is the noise level\n\nfor x_real in dataloader:\n    # 1. Add noise to create x_noisy\n    noise = torch.randn_like(x_real)\n    x_noisy = x_real + sigma * noise\n    \n    # 2. Compute the model score (gradient of the energy)\n    # Requires autograd.grad to compute the gradient of the scalar output (E_theta(x_noisy).sum()) with respect to the input (x_noisy)\n    model_score = -torch.autograd.grad(E_theta(x_noisy).sum(), x_noisy, create_graph=True)[0]\n    \n    # 3. Compute the score of the noisy data distribution (approximated by -noise/sigma)\n    data_score_approx = -noise / sigma\n    \n    # 4. Loss (MSE between the scores)\n    loss = torch.norm(model_score - data_score_approx, dim=-1).pow(2).mean()\n    \n    # 5. Backpropagation\n    loss.backward()\n    optimizer.step()\n```

## URL

https://arxiv.org/pdf/2103.04922
