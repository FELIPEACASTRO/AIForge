# Optuna - Hyperparameter Optimization Framework

## Description

Optuna is an open-source automatic hyperparameter optimization framework, designed specifically for machine learning. Its unique value proposition lies in its **imperative, define-by-run API**, which enables the construction of dynamic search spaces. This means that the structure of the search space can be defined by Python conditionals and loops, dynamically adapting to the results of previous trials. This flexibility sets it apart from traditional declarative approaches, enabling more complex and efficient optimizations.

## Statistics

**GitHub Stars:** Over 12,000 (as of Oct 2025);\n**Monthly Downloads (PyPI):** Over 7 million (as of Oct 2025);\n**Citations:** The original 2019 paper has over 10,000 citations, indicating wide adoption and impact in the scientific community.

## Features

Dynamic Search Spaces (Define-by-Run); Advanced Optimization Algorithms (including state-of-the-art samplers and pruning); Easy Parallelization (across threads or processes); Framework Agnosticism (compatible with PyTorch, TensorFlow, Scikit-learn, etc.); Optuna Dashboard (real-time web dashboard); LLM Integration (via Optuna MCP Server v4.4+); Multi-Objective Optimization (via Multi-Objective GPSampler v4.4+).

## Use Cases

Hyperparameter optimization for Machine Learning models (e.g., LightGBM, XGBoost, Neural Networks);\nFine-Tuning of Large Language Models (LLMs) with dynamic pruning;\nInnovative application in Feature Selection, treating inclusion/exclusion as a hyperparameter;\nIntegration in Predictive Maintenance and Scientific Research systems.

## Integration

**Installation:** `pip install optuna`\n\n**Basic Example (Function Minimization with Dynamic Space):**\n```python\nimport optuna\n\ndef objective(trial):\n    x = trial.suggest_float('x', -10, 10)\n    # define-by-run logic: Suggest 'y' only if 'x' is greater than 5\n    if x > 5:\n        y = trial.suggest_float('y', 0, 1)\n        return (x - 2)**2 + y\n    return (x - 2) ** 2\n\nstudy = optuna.create_study(direction='minimize')\nstudy.optimize(objective, n_trials=100)\n\nprint(f"Best value: {study.best_value}")\nprint(f"Best parameters: {study.best_params}")\n```\n\n**Pruning Example with an ML Framework:**\nOptuna uses `trial.report()` to report intermediate values and `trial.should_prune()` to decide whether to terminate an unpromising trial early, saving resources.

## URL

https://optuna.org/
