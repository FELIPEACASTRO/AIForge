# Energy-Based Models (EBMs) - Contrastive Divergence (CD) and Score Matching (SM)

## Description

Modelos Baseados em Energia (EBMs) são uma classe de modelos probabilísticos não normalizados que definem uma distribuição de probabilidade $p_\\theta(x) = \\frac{1}{Z(\\theta)} e^{-E_\\theta(x)}$, onde $E_\\theta(x)$ é a função de energia (tipicamente uma rede neural) e $Z(\\theta)$ é a função de partição intratável. A proposta de valor única dos EBMs reside na sua flexibilidade para modelar distribuições multimodais complexas, atribuindo baixa energia a dados reais e alta energia a dados irreais, sem as restrições arquitetônicas de GANs ou VAEs. **Divergência Contrastiva (CD)** e **Score Matching (SM)** são os dois principais métodos de treinamento para EBMs, cada um abordando o desafio da função de partição intratável de maneiras distintas.

## Statistics

CD minimiza a divergência KL ($D_{KL}$) entre a distribuição de dados e a distribuição do modelo, sendo conhecido por problemas de estabilidade (divergência de cadeia curta, CD-k). SM minimiza a divergência de Fisher ($D_F$), oferecendo uma função objetivo tratável e monitorável, sendo a base para os modernos Modelos Generativos Baseados em Score (como os modelos de difusão). Variantes como o Denoising Score Matching (DSM) e o Sliced Score Matching (SSM) melhoraram a estabilidade e a eficiência do treinamento em comparação com o CD tradicional.

## Features

EBMs: Modelagem de densidade de probabilidade implícita; Flexibilidade na arquitetura da função de energia; Capacidade inerente de detecção de anomalias (dados de alta energia). CD: Treinamento de máxima verossimilhança; Simples de implementar (CD-k); Requer amostragem (e.g., Dinâmica de Langevin) para estimar o gradiente. SM: Treinamento livre de amostragem (sampling-free); Objetivo de treinamento tratável e monitorável; Base para modelos de difusão.

## Use Cases

Geração de Conteúdo: Imagens, texto e áudio de alta qualidade. Detecção de Anomalias (Out-of-Distribution Detection): O valor de energia pode ser usado como uma métrica de anomalia. Aplicações em Sistemas de Energia: Previsão probabilística de geração de energia (eólica, solar). Otimização de Movimento: Integração como fatores de custo ou distribuições de amostragem iniciais em problemas de planejamento de movimento e robótica.

## Integration

A integração de EBMs é tipicamente realizada em frameworks de aprendizado profundo como PyTorch ou TensorFlow. O treinamento com CD envolve a simulação de cadeias de Markov (e.g., Dinâmica de Langevin) para obter amostras negativas. O treinamento com SM (especialmente DSM) é mais direto, focando em fazer o gradiente da função de energia (o score) corresponder ao score da distribuição de dados ruidosa. \n\n**Exemplo Conceitual de Treinamento com Score Matching (DSM):**\n```python\nimport torch\n\n# E_theta é a função de energia (rede neural)\n# sigma é o nível de ruído\n\nfor x_real in dataloader:\n    # 1. Adicionar ruído para criar x_noisy\n    noise = torch.randn_like(x_real)\n    x_noisy = x_real + sigma * noise\n    \n    # 2. Calcular o score do modelo (gradiente da energia)\n    # Requer autograd.grad para calcular o gradiente da saída escalar (E_theta(x_noisy).sum()) em relação à entrada (x_noisy)\n    model_score = -torch.autograd.grad(E_theta(x_noisy).sum(), x_noisy, create_graph=True)[0]\n    \n    # 3. Calcular o score da distribuição de dados ruidosa (aproximado por -noise/sigma)\n    data_score_approx = -noise / sigma\n    \n    # 4. Perda (MSE entre os scores)\n    loss = torch.norm(model_score - data_score_approx, dim=-1).pow(2).mean()\n    \n    # 5. Backpropagation\n    loss.backward()\n    optimizer.step()\n```

## URL

https://arxiv.org/pdf/2103.04922