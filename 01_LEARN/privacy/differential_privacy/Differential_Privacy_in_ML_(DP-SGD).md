# Differential Privacy in ML (DP-SGD)

## Description

**Differential Privacy (DP)** is a formal, mathematical framework for measuring and guaranteeing the privacy of individuals within a dataset. It ensures that the inclusion or exclusion of any single data record does not significantly affect the outcome of the trained model. The unique value proposition of DP is its **quantifiable and formal privacy guarantee**, expressed through a privacy budget ($\epsilon, \delta$), which allows for transparent control over the privacy-utility trade-off. **DP-SGD (Differentially-Private Stochastic Gradient Descent)** is the core algorithm that applies DP to Deep Learning by modifying the standard SGD process. / **Privacidade Diferencial (DP)** é um framework formal e matemático para medir e garantir a privacidade de indivíduos em um conjunto de dados. Garante que a inclusão ou exclusão de qualquer registro individual não afete significativamente o resultado do modelo treinado. A proposta de valor única da DP é sua **garantia de privacidade quantificável e formal**, expressa através de um orçamento de privacidade ($\epsilon, \delta$), que permite um controle transparente sobre o trade-off entre privacidade e utilidade. O **DP-SGD (Differentially-Private Stochastic Gradient Descent)** é o algoritmo central que aplica DP ao Deep Learning, modificando o processo padrão de SGD.

## Statistics

The privacy guarantee is quantified by the **Privacy Budget** ($\epsilon, \delta$). **Epsilon ($\epsilon$):** Measures the strength of the privacy guarantee; smaller values mean higher privacy. Practical applications often use $\epsilon$ values between 1 and 10. **Delta ($\delta$):** Represents the probability of the $\epsilon$-DP guarantee failing; typically set to a very small value (e.g., $10^{-5}$ or $10^{-7}$). **Utility Trade-off:** DP-SGD inevitably introduces a loss of model accuracy (utility) compared to standard SGD, a key challenge that requires careful hyperparameter tuning. The **Rényi Differential Privacy (RDP) Accountant** is the standard method used by libraries like Opacus and TensorFlow Privacy to track the cumulative $\epsilon$ consumed during training. / A garantia de privacidade é quantificada pelo **Orçamento de Privacidade** ($\epsilon, \delta$). **Epsilon ($\epsilon$):** Mede a força da garantia de privacidade; valores menores significam maior privacidade. Aplicações práticas usam $\epsilon$ entre 1 e 10. **Delta ($\delta$):** Probabilidade de falha da garantia $\epsilon$-DP; tipicamente um valor muito pequeno (e.g., $10^{-5}$). **Trade-off de Utilidade:** O DP-SGD introduz uma perda de precisão (utilidade) do modelo, um desafio que exige ajuste fino de hiperparâmetros. O **RDP Accountant** é o método padrão para rastrear o $\epsilon$ cumulativo consumido.

## Features

**Core Mechanisms (DP-SGD):** 1. **Per-Sample Gradient Computation:** Calculates gradients for each sample individually. 2. **Gradient Clipping:** Limits the L2 norm of each individual gradient to a maximum value $C$ to bound the influence of any single data point. 3. **Noise Addition:** Adds Gaussian noise to the clipped, averaged mini-batch gradient before updating model weights. **Capabilities:** Formal privacy guarantee, compatibility with complex Deep Learning models (CNNs, RNNs), and Privacy Accounting using the Rényi Differential Privacy (RDP) Accountant to track the consumed privacy budget ($\epsilon$) over training epochs. / **Mecanismos Centrais (DP-SGD):** 1. **Cálculo de Gradientes por Amostra:** Calcula gradientes para cada amostra individualmente. 2. **Clipping de Gradiente:** Limita a norma L2 de cada gradiente individual a um valor máximo $C$. 3. **Adição de Ruído:** Adiciona ruído Gaussiano ao gradiente médio do mini-lote. **Capacidades:** Garantia formal de privacidade, compatibilidade com modelos complexos de Deep Learning e Contabilidade de Privacidade usando o RDP Accountant.

## Use Cases

**Technology and Advertising:** Used by major tech companies (Google, Apple, Meta) for products like Google's **Community Mobility Reports** and **Ads prediction**, Apple's **Iconic Scenes** feature, and Meta's **Movement Range Maps**, ensuring private aggregation of user data. **Federated Learning:** Crucial for protecting model updates sent from local devices (e.g., smartphones) to a central server, providing **user-level differential privacy**. **Healthcare and Biomedicine:** Training ML models on sensitive patient data (e.g., electronic health records) for research while maintaining individual privacy. **Finance:** Analyzing financial transactions for fraud detection or risk modeling without exposing individual customer transaction data. / **Tecnologia e Publicidade:** Usado por grandes empresas (Google, Apple, Meta) em produtos como **Community Mobility Reports** e **previsão de anúncios** do Google, recurso **Iconic Scenes** da Apple e **Movement Range Maps** da Meta. **Aprendizado Federado:** Essencial para proteger as atualizações de modelo enviadas de dispositivos locais para um servidor central, fornecendo **privacidade diferencial no nível do usuário**. **Saúde e Biomedicina:** Treinamento de modelos de ML em dados sensíveis de pacientes. **Finanças:** Análise de transações financeiras para detecção de fraudes.

## Integration

DP-SGD is primarily integrated using specialized libraries for major ML frameworks. **Opacus (PyTorch):** Converts a standard PyTorch model into a DP-SGD model using the `PrivacyEngine.make_private()` method, requiring parameters like `noise_multiplier` ($\sigma$), `max_grad_norm` ($C$), and target $\epsilon$ and $\delta$. **TensorFlow Privacy:** Provides DP-SGD optimizers like `DPGradientDescentGaussianOptimizer` that replace standard optimizers in Keras models, requiring `l2_norm_clip` ($C$) and `noise_multiplier` ($\sigma$).

**Opacus (PyTorch) Example:**
```python
from opacus import PrivacyEngine
# ... model, optimizer, data_loader defined ...
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=data_loader,
    noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=5.0, target_delta=1e-5
)
# ... training loop ...
epsilon = privacy_engine.get_epsilon(target_delta=1e-5)
```
/ A integração é feita principalmente com bibliotecas especializadas. **Opacus (PyTorch):** Converte um modelo PyTorch padrão em DP-SGD usando `PrivacyEngine.make_private()`, exigindo parâmetros como `noise_multiplier` ($\sigma$), `max_grad_norm` ($C$) e $\epsilon$ e $\delta$ alvo. **TensorFlow Privacy:** Fornece otimizadores DP-SGD como `DPGradientDescentGaussianOptimizer` para modelos Keras.

**Exemplo Opacus (PyTorch):**
```python
from opacus import PrivacyEngine
# ... modelo, otimizador, data_loader definidos ...
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=data_loader,
    noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=5.0, target_delta=1e-5
)
# ... loop de treinamento ...
epsilon = privacy_engine.get_epsilon(target_delta=1e-5)
```

## URL

https://arxiv.org/abs/1607.00133 (Original DP-SGD Paper) | https://opacus.ai/ (Opacus - PyTorch) | https://www.tensorflow.org/responsible_ai/privacy/guide (TensorFlow Privacy)