# Optuna - Hyperparameter Optimization Framework

## Description

Optuna é um framework de otimização automática de hiperparâmetros de código aberto, projetado especificamente para aprendizado de máquina. Sua proposta de valor única reside na sua **API imperativa, define-by-run**, que permite a construção de espaços de busca dinâmicos. Isso significa que a estrutura do espaço de busca pode ser definida por condicionais e loops Python, adaptando-se dinamicamente aos resultados de testes anteriores. Essa flexibilidade o diferencia das abordagens declarativas tradicionais, permitindo otimizações mais complexas e eficientes.

## Statistics

**Estrelas no GitHub:** Mais de 12.000 (em out/2025);\n**Downloads Mensais (PyPI):** Mais de 7 milhões (em out/2025);\n**Citações:** O artigo original de 2019 possui mais de 10.000 citações, indicando ampla adoção e impacto na comunidade científica.

## Features

Espaços de Busca Dinâmicos (Define-by-Run); Algoritmos de Otimização Avançados (incluindo samplers de última geração e poda/pruning); Paralelização Fácil (em threads ou processos); Agnosticismo a Frameworks (compatível com PyTorch, TensorFlow, Scikit-learn, etc.); Optuna Dashboard (painel web em tempo real); Integração com LLM (via Optuna MCP Server v4.4+); Otimização Multiobjetivo (via Multi-Objective GPSampler v4.4+).

## Use Cases

Otimização de hiperparâmetros para modelos de Machine Learning (e.g., LightGBM, XGBoost, Redes Neurais);\nAjuste Fino (Fine-Tuning) de Large Language Models (LLMs) com poda dinâmica;\nAplicação inovadora em Seleção de Features, tratando a inclusão/exclusão como um hiperparâmetro;\nIntegração em sistemas de Manutenção Preditiva e Pesquisa Científica.

## Integration

**Instalação:** `pip install optuna`\n\n**Exemplo Básico (Minimização de Função com Espaço Dinâmico):**\n```python\nimport optuna\n\ndef objective(trial):\n    x = trial.suggest_float('x', -10, 10)\n    # Lógica define-by-run: Sugere 'y' apenas se 'x' for maior que 5\n    if x > 5:\n        y = trial.suggest_float('y', 0, 1)\n        return (x - 2)**2 + y\n    return (x - 2) ** 2\n\nstudy = optuna.create_study(direction='minimize')\nstudy.optimize(objective, n_trials=100)\n\nprint(f"Melhor valor: {study.best_value}")\nprint(f"Melhores parâmetros: {study.best_params}")\n```\n\n**Exemplo de Poda (Pruning) com Framework de ML:**\nO Optuna usa `trial.report()` para reportar valores intermediários e `trial.should_prune()` para decidir se deve encerrar um teste não promissor precocemente, economizando recursos.

## URL

https://optuna.org/