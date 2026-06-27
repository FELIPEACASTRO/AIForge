# Algoritmos de Otimização: Adam, AdamW, LAMB, Lion

## Description

Adam é um algoritmo de otimização de primeira ordem baseado em gradiente para funções objetivas estocásticas, combinando Momentum e RMSprop para taxas de aprendizado adaptativas. AdamW é uma variante que desacopla a regularização L2 (Weight Decay) da atualização do gradiente, melhorando a generalização. LAMB é otimizado para treinamento de grandes lotes, usando adaptação *layer-wise* para manter a estabilidade. Lion é um otimizador mais recente baseado em Sign Momentum Evoluído, oferecendo alta eficiência de memória e velocidade de convergência.

## Statistics

Adam: Hiperparâmetros $\\beta_1=0.9$, $\\beta_2=0.999$. AdamW: Desempenho superior ao Adam em tarefas com forte regularização. LAMB: Capacidade de lote de até 32.768+, reduzindo o tempo de treinamento do BERT de 3 dias para 76 minutos. Lion: Mais eficiente em memória que o Adam, converge mais rápido que o AdamW em benchmarks.

## Features

Adam: Taxas de aprendizado adaptativas, Correção de Vieses, Combinação de Momentum e RMSprop. AdamW: Decoupled Weight Decay, Melhor Generalização. LAMB: Adaptação Layer-wise, Otimização de Lotes Grandes. Lion: Atualização Baseada em Sinal, Eficiência de Memória, Menos Hiperparâmetros.

## Use Cases

Adam: Ampla gama de aplicações de Deep Learning (CNNs, GANs, VAEs). AdamW: Modelos de Transformadores (BERT, GPT, T5) e Vision Transformers (ViTs). LAMB: Pré-treinamento de Modelos de PNL em Grande Escala (BERT) e Treinamento de Modelos de Visão com Lotes Grandes. Lion: Grandes Arquiteturas de PNL (GPT-4) e Modelos de Visão Modernos.

## Integration

Adam: Incluído por padrão em todos os frameworks (PyTorch, TensorFlow, Keras). AdamW: Incluído no `torch.optim` do PyTorch. LAMB: Disponível em `tensorflow_addons` (TFA) ou bibliotecas de terceiros como `pytorch-optimizer`. Lion: Disponível em bibliotecas de terceiros como `lion-pytorch` ou `pytorch-optimizer`. Exemplos de código em Python/PyTorch e TensorFlow foram fornecidos.

## URL

https://arxiv.org/abs/1412.6980