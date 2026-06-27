# Aprendizado Federado Avançado: FedAvg, FedProx e Personalização

## Description

O Aprendizado Federado (FL) é um paradigma de aprendizado de máquina distribuído que permite que múltiplos clientes (dispositivos, hospitais, bancos) treinem um modelo compartilhado de forma colaborativa, mantendo os dados de treinamento localmente para preservar a privacidade. O **FedAvg (Federated Averaging)** é o algoritmo fundamental, que calcula a média ponderada das atualizações do modelo dos clientes. O **FedProx (Federated Proximal)** é uma extensão crucial do FedAvg que aborda a **heterogeneidade** (dados não-IID e sistemas variados) adicionando um termo de regularização proximal para estabilizar a convergência e mitigar o "desvio do cliente" (client drift). A **Personalização em FL** é a etapa avançada que adapta o modelo global para atender às necessidades específicas de cada cliente, melhorando o desempenho local em cenários altamente heterogêneos.

## Statistics

**FedProx em Heterogeneidade:** Em ambientes altamente heterogêneos, o FedProx demonstrou uma convergência significativamente mais estável e precisa em relação ao FedAvg, melhorando a precisão absoluta do teste em **22% em média** em um conjunto de dados federados realistas. **Personalização:** A personalização pode levar a um **aumento de desempenho de 5-10%** na precisão do modelo para clientes individuais em comparação com o modelo global não personalizado. **FedAvg:** Redução de até 10-100x nas rodadas de comunicação em comparação com o SGD distribuído tradicional.

## Features

**FedAvg:** Agregação ponderada de modelos, comunicação eficiente. **FedProx:** Regularização proximal (termo $\mu$) para estabilidade, tolerância à heterogeneidade de dados (não-IID) e de sistema (poder computacional/largura de banda). **Personalização:** Adaptação do modelo (e.g., fine-tuning local, modelos híbridos), otimização do desempenho em nível de cliente.

## Use Cases

**Saúde Personalizada:** Previsão de incapacidade ou segmentação de múltiplos órgãos abdominais, onde a distribuição de dados varia significativamente entre hospitais ou pacientes. O FedProx e a personalização garantem que o modelo adaptado localmente seja mais preciso para cada instituição. **Edge Computing/IoT:** Dispositivos como smartphones, sensores IoT e veículos autônomos, onde a heterogeneidade do sistema (poder computacional, largura de banda) e a heterogeneidade estatística (padrões de uso de dados) são altas. O FedProx garante a estabilidade do treinamento e a personalização melhora a experiência do usuário final. **Finanças:** Detecção de fraude entre bancos com diferentes perfis de clientes.

## Integration

**Frameworks:** Flower, TensorFlow Federated (TFF), PySyft.
**Exemplo Conceitual FedProx (Função de Perda):**
A função de perda local $F_k$ para o cliente $k$ é modificada para incluir o termo proximal:
$$F_k(\mathbf{w}) = f_k(\mathbf{w}) + \frac{\mu}{2} ||\mathbf{w} - \mathbf{w}^t||^2$$
Onde $\mathbf{w}^t$ é o modelo global da rodada anterior e $\mu$ é o parâmetro de regularização.

**Exemplo de Código (Flower Framework - FedProx Strategy):**
```python
from flwr.server.strategy import FedProx

# Configuração da estratégia FedProx
# O parâmetro `proximal_mu` corresponde ao termo de regularização μ
strategy = FedProx(
    fraction_fit=0.1,
    min_available_clients=10,
    proximal_mu=0.01  # O valor de mu (μ)
)

# Iniciar o servidor FL com a estratégia FedProx
# flwr.server.start_server(strategy=strategy, ...)
```

## URL

FedAvg: https://arxiv.org/abs/1602.05629 | FedProx: https://arxiv.org/abs/1812.06127 | Flower: https://flower.ai