# Ray Tune

## Description

Ray Tune is a Python library for running experiments and **scalable hyperparameter optimization** at any scale, built on top of the Ray distributed computing framework. Its unique value proposition lies in its ability to **transparently parallelize** hyperparameter search across multiple GPUs and nodes, enabling experiments to scale up to 100 times and reducing costs by up to 10 times through the use of preemptible instances. It offers a unified API for integrating state-of-the-art algorithms and connects seamlessly with the leading Machine Learning (ML) frameworks, removing the complexity of managing distributed systems for ML optimization.

## Statistics

**Scalability and Performance**: Enables experiments to scale up to **100 times** faster than single-node solutions. **Cost Optimization**: Can reduce compute costs by up to **10 times** by leveraging low-cost preemptible instances. **Adoption**: It is a key component of the Ray ecosystem, which has more than **35,000 stars** on GitHub, indicating strong adoption and community. **Scientific Paper**: Based on the paper "Tune: A Research Platform for Distributed Model Selection and Training" (arXiv:1807.05118), published in 2018.

## Features

**State-of-the-Art Optimization Algorithms**: Supports algorithms such as Population Based Training (PBT) and HyperBand/ASHA, in addition to integrating with external tools like Ax, BayesOpt, BOHB, Nevergrad, and Optuna. **Developer Productivity**: Enables model optimization with the addition of just a few lines of code, supporting multiple storage options for experiment results (NFS, cloud storage) and logging to tools such as MLflow and TensorBoard. **Distributed and Multi-GPU Training**: Offers transparent parallelization across multiple GPUs and multiple nodes, with fault tolerance and native support for cloud environments. **Integration with ML Frameworks**: Native compatibility with PyTorch, TensorFlow/Keras, XGBoost, scikit-learn, and others.

## Use Cases

**Large Language Model (LLM) Optimization**: Used to tune hyperparameters of transformer models and LLMs in distributed environments. **Computer Vision**: Optimization of convolutional neural network (CNN) architectures for tasks such as image classification and segmentation. **Reinforcement Learning (RL)**: Used in RL frameworks such as RLlib (also part of Ray) to optimize RL policies and algorithms. **Time Series Forecasting**: Optimization of complex forecasting models, such as NeuroCard, for cardinality estimation in databases. **ML Research**: A foundational platform for researchers seeking to compare and develop new hyperparameter optimization algorithms efficiently and at scale.

## Integration

Integration with Ray Tune is done by defining a training function (objective function) that accepts a dictionary of hyperparameter configuration and reports metrics through the `tune.report()` API. The `tune.Tuner` is then used to start the search, defining the search space (`param_space`) and the algorithm/scheduler.

**Integration Example with PyTorch (Conceptual):**
```python
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig

def train_model(config):
    # 1. Load data and model
    # 2. Define hyperparameters from 'config'
    lr = config["lr"]
    epochs = config["epochs"]
    
    for epoch in range(epochs):
        # Training logic
        loss = ...
        accuracy = ...
        
        # Report metrics to Ray Tune
        tune.report(loss=loss, accuracy=accuracy)

# Define the search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "epochs": tune.choice([5, 10, 15]),
}

# Configure the Tuner
tuner = Tuner(
    train_model,
    param_space=search_space,
    tune_config=TuneConfig(
        metric="accuracy",
        mode="max",
        num_samples=10, # Number of samples to test
        scheduler=tune.schedulers.ASHAScheduler(), # Example scheduler
    ),
    run_config=RunConfig(name="my_tune_experiment")
)

results = tuner.fit()
best_result = results.get_best_result()
print(f"Best Configuration: {best_result.config}")
```

## URL

https://docs.ray.io/en/latest/tune/index.html