# Técnicas de Regularização: Dropout e Decaimento de Peso (L2, AdamW)

## Description

As **Técnicas de Regularização** são métodos essenciais em Deep Learning para **prevenir o *overfitting*** e melhorar a **generalização** do modelo, reduzindo sua complexidade efetiva. O **Dropout** desativa aleatoriamente neurônios durante o treinamento para forçar a rede a aprender representações mais robustas e evitar a co-adaptação. Suas variantes incluem o **DropConnect**, que desativa conexões (pesos) em vez de neurônios, e o **Spatial Dropout**, que desativa canais inteiros em CNNs para regularizar camadas convolucionais. Os métodos de **Decaimento de Peso (Weight Decay)**, como a Regularização L2 ($\lambda \sum w^2$), adicionam uma penalidade à função de perda para incentivar pesos menores e modelos mais simples. O **AdamW** é uma implementação aprimorada do Decaimento de Peso L2 para otimizadores adaptativos (como Adam), desacoplando a penalidade da atualização do gradiente para garantir uma regularização mais correta e eficaz, sendo o padrão para o treinamento de grandes modelos de linguagem (LLMs) e *Transformers* [4].

## Statistics

**Taxa de Dropout ($p$):** Geralmente entre 0.1 e 0.5; 0.5 é o valor comum para camadas ocultas [1]. **Fator de Escala:** Ativações restantes são escaladas por $1/(1-p)$ durante o treinamento. **Hiperparâmetro $\lambda$ (Decaimento de Peso):** Valores típicos na faixa de $10^{-4}$ a $10^{-2}$. **Melhoria de Desempenho:** O Dropout pode reduzir a taxa de erro em tarefas de visão e fala em 1-3% [1]. O AdamW demonstrou melhor desempenho de generalização em modelos de *Transformer* em comparação com o Adam com L2 Regularization tradicional [4].

## Features

**Dropout e Variantes:** Desativação aleatória de neurônios (Dropout), desativação aleatória de conexões (DropConnect), desativação de canais inteiros em CNNs (Spatial Dropout). **Decaimento de Peso:** Penalidade L1 (seleção de *features*), Penalidade L2 (redução da magnitude dos pesos), AdamW (Decaimento de Peso desacoplado para otimizadores adaptativos). **Proposta de Valor Única:** Redução da co-adaptação de neurônios (Dropout), Simplificação do modelo e suavização da superfície de decisão (Weight Decay), Implementação correta e eficaz do Decaimento de Peso em otimizadores avançados (AdamW).

## Use Cases

**Visão Computacional (CNNs):** Uso de **Spatial Dropout** em arquiteturas como ResNet e VGG. **Processamento de Linguagem Natural (NLP):** Uso de **Dropout** em modelos de RNNs, LSTMs e, especialmente, em modelos de **Transformer** (BERT, GPT) para regularizar camadas de atenção. **Modelos de Recomendação:** Aplicação de Dropout e DropConnect para evitar memorização de interações. **Modelos de Grande Escala (LLMs):** O otimizador **AdamW** é o padrão *de facto* para o treinamento de LLMs e modelos de visão devido à sua generalização superior [4].

## Integration

**TensorFlow/Keras:** Uso da camada `tf.keras.layers.Dropout` e do argumento `kernel_regularizer=l2(lambda)` nas camadas `Dense`. **PyTorch (Spatial Dropout):** Uso de `nn.Dropout2d` para desativar canais em camadas convolucionais. **PyTorch (AdamW):** Uso do otimizador `torch.optim.AdamW` com o parâmetro `weight_decay` para aplicar o decaimento de peso de forma desacoplada e correta.
```python
# Exemplo AdamW (PyTorch)
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
```python
# Exemplo Dropout e L2 (Keras)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
model.add(Dense(128, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
```

## URL

http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf