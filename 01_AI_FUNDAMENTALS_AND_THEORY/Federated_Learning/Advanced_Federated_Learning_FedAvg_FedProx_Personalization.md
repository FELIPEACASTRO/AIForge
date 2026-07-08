# Advanced Federated Learning: FedAvg, FedProx, and Personalization

## Description

Federated Learning (FL) is a distributed machine learning paradigm that enables multiple clients (devices, hospitals, banks) to collaboratively train a shared model while keeping the training data local to preserve privacy. **FedAvg (Federated Averaging)** is the foundational algorithm, which computes the weighted average of the clients' model updates. **FedProx (Federated Proximal)** is a crucial extension of FedAvg that addresses **heterogeneity** (non-IID data and diverse systems) by adding a proximal regularization term to stabilize convergence and mitigate "client drift." **Personalization in FL** is the advanced stage that adapts the global model to meet the specific needs of each client, improving local performance in highly heterogeneous scenarios.

## Statistics

**FedProx under Heterogeneity:** In highly heterogeneous environments, FedProx has demonstrated significantly more stable and accurate convergence compared to FedAvg, improving absolute test accuracy by **22% on average** on a realistic set of federated datasets. **Personalization:** Personalization can lead to a **performance boost of 5-10%** in model accuracy for individual clients compared to the non-personalized global model. **FedAvg:** Up to a 10-100x reduction in communication rounds compared to traditional distributed SGD.

## Features

**FedAvg:** Weighted model aggregation, efficient communication. **FedProx:** Proximal regularization (the $\mu$ term) for stability, tolerance to data (non-IID) and system (compute power/bandwidth) heterogeneity. **Personalization:** Model adaptation (e.g., local fine-tuning, hybrid models), client-level performance optimization.

## Use Cases

**Personalized Healthcare:** Disability prediction or multi-organ abdominal segmentation, where the data distribution varies significantly across hospitals or patients. FedProx and personalization ensure that the locally adapted model is more accurate for each institution. **Edge Computing/IoT:** Devices such as smartphones, IoT sensors, and autonomous vehicles, where system heterogeneity (compute power, bandwidth) and statistical heterogeneity (data usage patterns) are high. FedProx ensures training stability and personalization improves the end-user experience. **Finance:** Fraud detection across banks with different customer profiles.

## Integration

**Frameworks:** Flower, TensorFlow Federated (TFF), PySyft.
**Conceptual FedProx Example (Loss Function):**
The local loss function $F_k$ for client $k$ is modified to include the proximal term:
$$F_k(\mathbf{w}) = f_k(\mathbf{w}) + \frac{\mu}{2} ||\mathbf{w} - \mathbf{w}^t||^2$$
Where $\mathbf{w}^t$ is the global model from the previous round and $\mu$ is the regularization parameter.

**Code Example (Flower Framework - FedProx Strategy):**
```python
from flwr.server.strategy import FedProx

# FedProx strategy configuration
# The `proximal_mu` parameter corresponds to the regularization term μ
strategy = FedProx(
    fraction_fit=0.1,
    min_available_clients=10,
    proximal_mu=0.01  # The value of mu (μ)
)

# Start the FL server with the FedProx strategy
# flwr.server.start_server(strategy=strategy, ...)
```

## URL

FedAvg: https://arxiv.org/abs/1602.05629 | FedProx: https://arxiv.org/abs/1812.06127 | Flower: https://flower.ai
