# Temporal Graph Networks (TGN)

## Description

Temporal Graph Networks (TGNs) are a generic and efficient framework for deep learning on dynamic graphs, represented as sequences of timed events. TGN addresses the limitation of traditional Graph Neural Networks (GNNs), which cannot effectively model the evolving nature of many real-world systems (such as social networks, financial transactions, and biological interactions). Its unique value proposition lies in combining **memory modules** and **graph-based operators** to generate temporal node embeddings that capture the state of each node at any point in time, allowing the model to adapt to and predict changes in the graph's structure and features over time. TGN generalizes several previous dynamic graph learning models, such as JODIE and DyRep, as specific instances of its framework.

## Statistics

**State-of-the-Art (SOTA) Performance:** TGN achieved SOTA performance on several transductive and inductive prediction tasks on dynamic graphs at the time of its publication. **Efficiency:** It was shown to be more computationally efficient than previous models like JODIE and DyRep. **Citations:** The original paper (`arXiv:2006.10637`) has over 1,100 citations (as of 2024), indicating its wide adoption and influence in dynamic graph learning research. **Implementation:** The official GitHub repository has over 1,100 stars and 220 forks, reflecting its popularity in the research community. **Language:** Implemented 100% in Python, using the PyTorch library.

## Features

**Memory Modules:** Store and update a memory state for each node, allowing the model to retain information about the node's interaction history. The memory is updated after each temporal event (interaction). **Message Aggregation Function:** Aggregates messages from temporal neighbors to form a new message for the target node. TGN supports different aggregators, such as attention aggregation (TGN-attn) and sum aggregation. **Temporal Embedding Module:** Encodes the time difference between the current event and the last event stored in memory, allowing the model to capture the importance of recency. **Generalization:** The TGN framework is flexible and can be configured to replicate the behavior of previous dynamic graph models, such as JODIE and DyRep, through different configurations of its components. **Computational Efficiency:** Designed to be more computationally efficient than previous approaches, especially on transductive and inductive tasks.

## Use Cases

**Link Prediction:** Predict the formation of new edges (interactions) in a dynamic graph, such as predicting future connections in social networks or transactions in financial networks. **Node Classification:** Classify the type or state of a node at a given moment, such as identifying malicious users in a social network or classifying the entity type in an evolving knowledge graph. **Graph Anomaly Detection:** Identify unusual events or nodes that deviate from the graph's normal behavior, such as fraud detection in financial transactions or identifying attacks in communication networks. **Recommendation Systems:** Model the temporal interactions between users and items to provide more accurate and time-sensitive recommendations. **Physical Systems Modeling:** Predict the long-term dynamics of complex physical systems, such as in physics simulations.

## Integration

The most popular and maintained implementation of TGN is available in **PyTorch Geometric (PyG)**, a leading library for GNNs in PyTorch. The `torch_geometric.nn.models.TGN` class provides a ready-to-use implementation.

**Integration Example (PyTorch Geometric):**

```python
import torch
from torch_geometric.nn import TGN
from torch_geometric.datasets import TGNExample

# 1. Load an example dynamic graph dataset
dataset = TGNExample(name='wikipedia')
data = dataset[0]

# 2. Initialize the TGN model
# Key parameters:
# - num_nodes: Total number of nodes
# - raw_msg_dim: Dimension of the message feature (usually the edge feature dimension)
# - memory_dim: Dimension of each node's memory state
# - time_dim: Dimension of the temporal embedding
# - num_layers: Number of graph attention layers
# - use_memory: Whether to use the memory module
# - aggregator: Type of message aggregator ('last', 'mean', 'attn')

tgn = TGN(
    num_nodes=data.num_nodes,
    raw_msg_dim=data.msg.size(-1),
    memory_dim=100,
    time_dim=100,
    num_layers=1,
    use_memory=True,
    aggregator='attn'
)

# 3. Usage example (Link Prediction)
# TGN is typically used in a training loop that processes events in batches.

# Simulation of a batch of events (interactions)
src = data.src[:100]
dst = data.dst[:100]
t = data.t[:100]
msg = data.msg[:100]

# 4. Obtain temporal node embeddings
# TGN returns the node embeddings *before* and *after* the memory update.
# The memory module is updated internally.
z, last_update = tgn(src, dst, t, msg)

# z contains the node embeddings for src and dst.
# last_update contains the time of the last update for each node.

# 5. Reset the memory state (required before a new epoch or test)
# tgn.memory.reset_state()
```

**Dependencies:**
*   `torch`
*   `torch-geometric` (PyG)
*   `pandas`
*   `scikit-learn`

## URL

https://github.com/twitter-research/tgn