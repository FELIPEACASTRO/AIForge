# Randomized Smoothing (Suavização Aleatória)

## Description

A **Certified Defense** is a technique in **Robust Machine Learning** that provides a mathematically guaranteed lower bound on the size of the adversarial perturbation required to change a model's prediction. Unlike empirical defenses, which can be bypassed by stronger attacks, certified defenses offer a provable guarantee of robustness within a specific radius (e.g., $l_2$-norm) around a data point. **Randomized Smoothing (RS)** is the most prominent and practical method for achieving this certified robustness, particularly for large-scale models and datasets like ImageNet. The unique value proposition of RS is its simplicity and scalability: it transforms any base classifier into a new, smoothed classifier that is certifiably robust by classifying the input based on the most probable class after adding random noise (typically Gaussian) to the input multiple times. This technique allows for the calculation of a provable robustness radius around each input, a significant advancement over previous methods.

## Statistics

**Artigo Seminal (Cohen et al., 2019)**: O artigo "Certified Adversarial Robustness via Randomized Smoothing" é um dos mais citados na área de robustez certificada, com mais de 2.700 citações (em 2024), destacando sua influência. **Escalabilidade**: Foi o primeiro método a demonstrar robustez certificada em grande escala no conjunto de dados **ImageNet**, com raios de robustez significativos. **Raio de Robustez**: O raio de robustez ($r$) é diretamente proporcional ao desvio padrão ($\sigma$) do ruído Gaussiano usado, e inversamente relacionado à probabilidade de erro. A fórmula fundamental para o raio certificado é $r = \sigma \Phi^{-1}(p_A)$, onde $\Phi^{-1}$ é a função inversa da CDF Gaussiana e $p_A$ é o limite inferior da probabilidade da classe mais provável. **Desempenho**: Embora forneça garantias, os modelos suavizados tendem a ter uma precisão de classificação ligeiramente menor em entradas limpas (não adversárias) em comparação com modelos treinados empiricamente. Por exemplo, em ImageNet, a precisão certificada pode ser de 40-50% para um raio $l_2$ de 0.5, enquanto a precisão limpa pode ser de 60-70%.

## Features

**Transformação de Classificador Base**: Converte qualquer classificador em um classificador suavizado com robustez certificada. **Robustez Certificada Provável**: Oferece uma garantia matemática de que o modelo não mudará sua previsão dentro de um raio de perturbação específico ($l_2$-norm). **Escalabilidade**: Demonstrou ser eficaz em grandes conjuntos de dados e redes neurais profundas, como o ImageNet. **Simplicidade**: A técnica é relativamente simples de implementar, envolvendo a adição de ruído aleatório (geralmente Gaussiano) à entrada e a votação da classe mais provável. **Generalidade**: Não está restrito a um tipo específico de ataque adversário, fornecendo uma defesa certificada contra qualquer ataque dentro do raio especificado.

## Use Cases

**Sistemas de Missão Crítica**: Aplicações onde a segurança e a confiabilidade são primordiais, como veículos autônomos, diagnóstico médico por imagem e sistemas de controle industrial. A garantia certificada impede que pequenas perturbações causem falhas catastróficas. **Detecção de Fraude Financeira**: Em modelos de detecção de anomalias, a suavização aleatória pode garantir que um invasor não consiga enganar o modelo com pequenas alterações nos dados de transação para evitar a detecção. **Classificação de Imagens de Alta Segurança**: Em cenários como reconhecimento facial ou vigilância, onde a manipulação de imagens de entrada pode ter sérias consequências. **Plataformas de Machine Learning como Serviço (MLaaS)**: Provedores de serviços de ML podem usar a suavização aleatória para oferecer um nível de serviço com garantia de robustez, diferenciando-se de defesas empíricas. **Pesquisa em Robustez de IA**: Serve como um ponto de referência fundamental (baseline) para o desenvolvimento e comparação de novas técnicas de defesa certificada.

## Integration

A implementação de Suavização Aleatória envolve duas etapas principais: treinamento do classificador base e certificação do classificador suavizado. O treinamento é feito com ruído, e a certificação é um processo estatístico.

**Exemplo de Certificação (Python/PyTorch - Conceitual):**

```python
import torch
import numpy as np
from scipy.stats import norm

# Parâmetros
sigma = 0.25  # Desvio padrão do ruído Gaussiano
num_samples = 1000  # Número de amostras para estimar a classe suavizada
alpha = 0.001  # Nível de significância para o certificado

def certify(x, base_classifier, sigma, num_samples, alpha):
    """
    Certifica a robustez de um ponto de dados 'x' usando Suavização Aleatória.
    """
    counts = {}
    for _ in range(num_samples):
        # 1. Adicionar ruído Gaussiano
        noise = sigma * torch.randn_like(x)
        x_noisy = x + noise
        
        # 2. Classificar a amostra ruidosa
        with torch.no_grad():
            prediction = base_classifier(x_noisy).argmax().item()
        
        counts[prediction] = counts.get(prediction, 0) + 1

    # Classe mais votada
    c_A = max(counts, key=counts.get)
    n_A = counts[c_A]
    
    # Estimar o limite inferior para a probabilidade de c_A (p_A)
    # Usando o limite inferior de Clopper-Pearson (simplificado por Hoeffding/Chernoff)
    # O cálculo exato do raio requer a função inversa da CDF Gaussiana (norm.ppf)
    
    # Cálculo do raio certificado (r)
    # p_A_lower é o limite inferior da probabilidade de c_A
    # O cálculo exato é complexo, mas o princípio é:
    # r = sigma * norm.ppf(p_A_lower)
    
    # Exemplo simplificado para fins didáticos:
    if n_A / num_samples > 0.5:
        # A fórmula real usa o limite inferior de Clopper-Pearson para p_A
        # e depois a função inversa da CDF Gaussiana para obter o raio.
        # Aqui, apenas um placeholder conceitual.
        p_A_lower = 0.5 # Placeholder
        radius = sigma * norm.ppf(p_A_lower)
        return c_A, radius
    else:
        return None, 0.0

# NOTA: A implementação real requer a biblioteca `smooth-certify` ou similar
# para os cálculos estatísticos precisos.
```

**Dependências Comuns:** `PyTorch`, `NumPy`, `SciPy` (para funções estatísticas).

**URL para Implementação de Referência:** O código-fonte do artigo seminal (Cohen et al., 2019) é frequentemente usado como referência.
**Instalação:** Geralmente requer a instalação de um pacote específico de suavização aleatória ou a implementação manual com bibliotecas padrão de ML.
`pip install smooth-certify` (exemplo de pacote de terceiros).

## URL

https://arxiv.org/abs/1902.02918