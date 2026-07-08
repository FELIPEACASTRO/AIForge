# Neptune.ai

## Description

Neptune.ai is an **experiment tracker and metadata store for MLOps**, designed specifically for training and debugging **foundation models** at large scale. Its unique value proposition lies in the ability to monitor thousands of per-layer metrics — losses, gradients, and activations — at any scale, with lag-free visualization and 100% accurate rendering, even with millions of data points. It focuses on being a dedicated and non-intrusive tool, integrating easily with existing ML stacks, and offers SaaS or self-hosted deployment options via a Helm chart for Kubernetes.

## Statistics

**Users:** More than **60,000 AI researchers** and **1,500 commercial and research teams** use the platform. **Projects:** More than **30,000 projects** tracked. **Recognition:** Included in CB Insights' "Top 100 AI Startups" list in 2021 and 2022. **Rating:** A rating of **4.8 out of 5 stars** in user reviews (G2). **Funding:** Backed by **$18 million** in funding. **Data Scale:** Public demonstrations with more than **100 million data points** per run.

## Features

**Scalable Experiment Tracking:** Log, display, organize, and compare ML experiments in a single place, optimized for foundation models and LLMs. **Centralized Metadata Store:** Storage of MLOps metadata, including scripts, data, parameters, and metrics. **Deep Debugging:** Facilitates the identification of training problems (exploding/vanishing gradients, convergence failures) by isolating issues in specific layers. **Forking of Runs:** Allows testing multiple configurations simultaneously and branching from the best final step, maintaining the complete experiment lineage. **Model Registry:** Functionality to version, store, and organize model metadata. **High-Performance Visualization:** Fast data filtering and searching, visualization, and comparison of metrics, parameters, and learning curves in real time.

## Use Cases

**Foundation Model Training:** Monitoring and debugging large language models (LLMs) and other foundation models at massive scale (e.g., OpenAI, Bioptimus). **Hyperparameter Optimization:** Grouping, filtering, and ranking thousands of experiments for clear insights and confident decision-making. **AI Research and Development:** Provides a transparent and searchable record of the work, essential for rigorous applied science (e.g., KoBold Metals). **MLOps and Model Management:** Integration into MLOps pipelines for structured management and enhanced security (e.g., Veo Technologies, Cradle). **Notable Companies:** OpenAI, Samsung, Roche, HP, Brainly, Ginkgo Bioworks, Cradle, KoBold Metals, InstaDeep, Bioptimus.

## Integration

Neptune.ai integrates with several machine learning libraries (PyTorch, TensorFlow, Keras, Scikit-learn, etc.) through loggers or callbacks. The primary usage involves initializing a "run" and logging configurations and metrics. The integration uses the `neptune-scale` library for logging and `neptune-query` for querying.

**Integration Example (Python):**

```python
# 1. Install the libraries
# pip install neptune-scale neptune-query

# 2. Connect and Create a Run (neptune-scale)
from neptune_scale import Run

run = Run(
    run_id="MY-PROJECT-123", # Project and run ID
    experiment_name="foundation-model-training"
)

# 3. Log Hyperparameters and the Training Process
run.log_configs(
    {
        "params/learning_rate": 0.001,
        "params/optimizer": "Adam",
    }
)

# Training loop
for step in range(100):
    # Simulate metric logging
    accuracy = 0.87 + (step / 1000)
    loss = 0.14 - (step / 5000)
    
    run.log_metrics(
        data={
            "train/accuracy": accuracy,
            "train/loss": loss,
        },
        step=step,
    )

# 4. Query Logs for Analysis (neptune-query)
import neptune_query as nq

# Fetch metadata as a table (DataFrame)
# table = nq.fetch_experiments_table(
#     experiments=r"foundation-model-.*",
#     attributes=r".*metric.*/val_.+",
# )
# print(table.head())

# Stop the run
run.stop()
```

## URL

https://neptune.ai/