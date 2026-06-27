# DARTS (Differentiable Architecture Search)

## Description

O DARTS (Differentiable Architecture Search) é um método de Neural Architecture Search (NAS) que aborda o desafio de escalabilidade ao formular a busca de arquitetura de forma diferenciável. Ele relaxa a busca discreta em um espaço de busca contínuo, permitindo o uso de otimização baseada em gradiente (descida de gradiente) para encontrar a melhor arquitetura. Isso reduz drasticamente o custo computacional da busca em comparação com métodos de otimização não diferenciáveis.

## Statistics

Redução significativa no custo de busca (horas de GPU). Resultados de ponta em tarefas como classificação de imagens (e.g., 94.36% de acurácia no CIFAR-10). O artigo original foi amplamente citado (mais de 6.000 citações).

## Features

Relaxamento Contínuo do Espaço de Busca; Otimização Baseada em Gradiente (Descida de Gradiente); Compartilhamento de Pesos (Weight Sharing) dentro de uma super-rede; Busca por 'Células' (blocos de construção) que são então empilhadas; Uso de otimização de segunda ordem.

## Use Cases

Classificação de Imagens (CIFAR-10, ImageNet); Modelos Recorrentes (RNNs); Ponto de partida para métodos NAS mais avançados (e.g., DARTS de segunda ordem, P-DARTS, PC-DARTS) e aplicações em visão computacional e processamento de linguagem natural.

## Integration

A integração geralmente envolve a implementação da super-rede e do algoritmo de otimização de dois níveis (pesos e arquitetura). Exemplo de código (conceitual em PyTorch):\n```python\n# Instalação (exemplo de implementação popular)\n# pip install darts-nas\n\n# Exemplo conceitual de otimização de dois níveis\n# Otimizador para pesos do modelo (w)\noptimizer_w = torch.optim.SGD(model.weights(), lr=LR_W)\n# Otimizador para parâmetros de arquitetura (alpha)\noptimizer_a = torch.optim.Adam(model.alphas(), lr=LR_A, betas=(0.5, 0.999), weight_decay=WEIGHT_DECAY)\n\n# Loop de treinamento\nfor step in range(num_steps):\n    # 1. Otimizar pesos (w) usando o conjunto de treinamento\n    loss_w = model(input_train, target_train)\n    optimizer_w.zero_grad()\n    loss_w.backward()\n    optimizer_w.step()\n\n    # 2. Otimizar arquitetura (alpha) usando o conjunto de validação\n    loss_a = model(input_valid, target_valid)\n    optimizer_a.zero_grad()\n    loss_a.backward()\n    optimizer_a.step()\n```

## URL

https://arxiv.org/abs/1806.09055