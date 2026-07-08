# Randomized Smoothing

## Description

A **Certified Defense** is a technique in **Robust Machine Learning** that provides a mathematically guaranteed lower bound on the size of the adversarial perturbation required to change a model's prediction. Unlike empirical defenses, which can be bypassed by stronger attacks, certified defenses offer a provable guarantee of robustness within a specific radius (e.g., $l_2$-norm) around a data point. **Randomized Smoothing (RS)** is the most prominent and practical method for achieving this certified robustness, particularly for large-scale models and datasets like ImageNet. The unique value proposition of RS is its simplicity and scalability: it transforms any base classifier into a new, smoothed classifier that is certifiably robust by classifying the input based on the most probable class after adding random noise (typically Gaussian) to the input multiple times. This technique allows for the calculation of a provable robustness radius around each input, a significant advancement over previous methods.

## Statistics

**Seminal Paper (Cohen et al., 2019)**: The paper "Certified Adversarial Robustness via Randomized Smoothing" is one of the most cited in the field of certified robustness, with more than 2,700 citations (as of 2024), highlighting its influence. **Scalability**: It was the first method to demonstrate certified robustness at large scale on the **ImageNet** dataset, with significant robustness radii. **Robustness Radius**: The robustness radius ($r$) is directly proportional to the standard deviation ($\sigma$) of the Gaussian noise used, and inversely related to the error probability. The fundamental formula for the certified radius is $r = \sigma \Phi^{-1}(p_A)$, where $\Phi^{-1}$ is the inverse Gaussian CDF function and $p_A$ is the lower bound on the probability of the most probable class. **Performance**: Although it provides guarantees, smoothed models tend to have slightly lower classification accuracy on clean (non-adversarial) inputs compared to empirically trained models. For example, on ImageNet, the certified accuracy can be 40-50% for an $l_2$ radius of 0.5, while the clean accuracy can be 60-70%.

## Features

**Base Classifier Transformation**: Converts any classifier into a smoothed classifier with certified robustness. **Provable Certified Robustness**: Offers a mathematical guarantee that the model will not change its prediction within a specific perturbation radius ($l_2$-norm). **Scalability**: Has proven effective on large datasets and deep neural networks, such as ImageNet. **Simplicity**: The technique is relatively simple to implement, involving the addition of random noise (usually Gaussian) to the input and voting for the most probable class. **Generality**: It is not restricted to a specific type of adversarial attack, providing a certified defense against any attack within the specified radius.

## Use Cases

**Mission-Critical Systems**: Applications where safety and reliability are paramount, such as autonomous vehicles, medical image diagnosis, and industrial control systems. The certified guarantee prevents small perturbations from causing catastrophic failures. **Financial Fraud Detection**: In anomaly detection models, randomized smoothing can ensure that an attacker cannot fool the model with small changes to transaction data to avoid detection. **High-Security Image Classification**: In scenarios such as facial recognition or surveillance, where manipulation of input images can have serious consequences. **Machine Learning as a Service (MLaaS) Platforms**: ML service providers can use randomized smoothing to offer a service level with a robustness guarantee, differentiating themselves from empirical defenses. **AI Robustness Research**: Serves as a fundamental baseline for the development and comparison of new certified defense techniques.

## Integration

The implementation of Randomized Smoothing involves two main steps: training the base classifier and certifying the smoothed classifier. Training is done with noise, and certification is a statistical process.

**Certification Example (Python/PyTorch - Conceptual):**

```python
import torch
import numpy as np
from scipy.stats import norm

# Parameters
sigma = 0.25  # Standard deviation of the Gaussian noise
num_samples = 1000  # Number of samples to estimate the smoothed class
alpha = 0.001  # Significance level for the certificate

def certify(x, base_classifier, sigma, num_samples, alpha):
    """
    Certifies the robustness of a data point 'x' using Randomized Smoothing.
    """
    counts = {}
    for _ in range(num_samples):
        # 1. Add Gaussian noise
        noise = sigma * torch.randn_like(x)
        x_noisy = x + noise
        
        # 2. Classify the noisy sample
        with torch.no_grad():
            prediction = base_classifier(x_noisy).argmax().item()
        
        counts[prediction] = counts.get(prediction, 0) + 1

    # Most-voted class
    c_A = max(counts, key=counts.get)
    n_A = counts[c_A]
    
    # Estimate the lower bound for the probability of c_A (p_A)
    # Using the Clopper-Pearson lower bound (simplified via Hoeffding/Chernoff)
    # The exact radius computation requires the inverse Gaussian CDF function (norm.ppf)
    
    # Certified radius computation (r)
    # p_A_lower is the lower bound on the probability of c_A
    # The exact computation is complex, but the principle is:
    # r = sigma * norm.ppf(p_A_lower)
    
    # Simplified example for didactic purposes:
    if n_A / num_samples > 0.5:
        # The real formula uses the Clopper-Pearson lower bound for p_A
        # and then the inverse Gaussian CDF function to obtain the radius.
        # Here, just a conceptual placeholder.
        p_A_lower = 0.5 # Placeholder
        radius = sigma * norm.ppf(p_A_lower)
        return c_A, radius
    else:
        return None, 0.0

# NOTE: The real implementation requires the `smooth-certify` library or similar
# for the precise statistical computations.
```

**Common Dependencies:** `PyTorch`, `NumPy`, `SciPy` (for statistical functions).

**Reference Implementation URL:** The source code of the seminal paper (Cohen et al., 2019) is frequently used as a reference.
**Installation:** Generally requires the installation of a specific randomized smoothing package or a manual implementation with standard ML libraries.
`pip install smooth-certify` (example of a third-party package).

## URL

https://arxiv.org/abs/1902.02918
