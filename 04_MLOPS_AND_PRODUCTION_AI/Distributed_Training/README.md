# Distributed Training

> Distributed training splits a model and/or its data across many GPUs and nodes — using data, tensor, pipeline, and sharded parallelism plus collective communication — so models too large or too slow for a single accelerator can be trained efficiently.

## Why it matters

Modern foundation models exceed the memory and compute of any single GPU, so production teams must shard parameters, gradients, and optimizer states across hundreds or thousands of devices. The choice of parallelism strategy (and how it is composed into 3D/4D parallelism) directly drives training cost, wall-clock time, and the maximum trainable model size. Getting collective communication (NCCL), scheduling (Slurm/Kubernetes), and fault tolerance right is foundational MLOps for anyone training or fine-tuning large models.

## Taxonomy

| Approach | What is split | Trade-off | Canonical implementation |
|---|---|---|---|
| **Data Parallelism (DDP)** | The batch (full model replicated) | Simple, but full model must fit per GPU | [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html) |
| **Sharded Data Parallelism (ZeRO / FSDP)** | Optimizer states → gradients → params | Cuts memory ~Nx; more communication | [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) / [ZeRO](https://www.deepspeed.ai/tutorials/zero/) |
| **Tensor (Model) Parallelism** | Individual layer weight matrices | Low latency, needs fast intra-node links | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| **Pipeline Parallelism** | The model by layer/stage | Scales across nodes; "bubble" overhead | [GPipe](https://arxiv.org/abs/1811.06965) / [PipeDream](https://arxiv.org/abs/1806.03377) |
| **Sequence / Context Parallelism** | The sequence (activations) | Enables long context; pairs with TP | [Megatron sequence parallel](https://arxiv.org/abs/2205.05198) |
| **Expert Parallelism (MoE)** | Experts across devices | Sparse compute; all-to-all heavy | [DeepSpeed-MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/) |
| **3D / 4D Parallelism** | Compose DP × TP × PP (× SP/CP) | Max scale; complex to tune | [Megatron + DeepSpeed](https://arxiv.org/abs/2201.11990) |

## Key frameworks & tools

| Tool | Role | Link |
|---|---|---|
| NVIDIA Megatron-LM | Tensor/pipeline/sequence parallelism reference for LLMs | https://github.com/NVIDIA/Megatron-LM |
| Microsoft DeepSpeed | ZeRO, ZeRO-Infinity/Offload, MoE, pipeline engine | https://github.com/microsoft/DeepSpeed |
| PyTorch FSDP / FSDP2 | Native fully sharded data parallel (DTensor-based) | https://docs.pytorch.org/docs/stable/fsdp.html |
| PyTorch TorchTitan | Native one-stop LLM pre-training (FSDP2 + TP + PP) | https://github.com/pytorch/torchtitan |
| Hugging Face Accelerate | Unified launcher over FSDP/DeepSpeed, fp8, mixed precision | https://github.com/huggingface/accelerate |
| Megatron-Core / NeMo | Productionized Megatron building blocks | https://github.com/NVIDIA/NeMo |
| Ray Train | Cluster-level orchestration of DDP/FSDP/Accelerate jobs | https://docs.ray.io/en/latest/train/train.html |
| Horovod | All-reduce DP for TF/Keras/PyTorch | https://github.com/horovod/horovod |
| ColossalAI | Heterogeneous + auto parallelism toolkit | https://github.com/hpcaitech/ColossalAI |
| NVIDIA NCCL | GPU collective communication (all-reduce/all-gather) | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html |

### Scheduling & infrastructure

| Tool | Role | Link |
|---|---|---|
| Kubeflow Training Operator | PyTorchJob / MPIJob / TFJob CRDs on K8s | https://github.com/kubeflow/training-operator |
| Volcano | Batch / gang scheduling for distributed jobs on K8s | https://github.com/volcano-sh/volcano |
| Slurm | HPC workload manager for multi-node GPU clusters | https://slurm.schedmd.com/quickstart.html |
| NVIDIA NeMo-Run / Megatron launcher | Recipe-driven multi-node launch | https://github.com/NVIDIA/NeMo-Run |
| torchrun (elastic) | Native fault-tolerant/elastic launcher | https://docs.pytorch.org/docs/stable/elastic/run.html |

## Benchmarks & reference scales

| Reference | Scale demonstrated | Link |
|---|---|---|
| Megatron-Turing NLG 530B | 530B params, DP×TP×PP 3D parallelism | https://arxiv.org/abs/2201.11990 |
| ZeRO / ZeRO-Infinity | 100B+ on 400 GPUs; trillion-param feasibility | https://arxiv.org/abs/1910.02054 |
| Efficient large-scale training (Megatron) | 1T-param config, scaling efficiency on thousands of GPUs | https://arxiv.org/abs/2104.04473 |
| MLPerf Training | Standardized cross-vendor training benchmark | https://mlcommons.org/benchmarks/training/ |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism | 2019 | https://arxiv.org/abs/1909.08053 |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 2019 | https://arxiv.org/abs/1910.02054 |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | 2018 | https://arxiv.org/abs/1811.06965 |
| PipeDream: Fast and Efficient Pipeline Parallel DNN Training | 2018 | https://arxiv.org/abs/1806.03377 |
| Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | 2021 | https://arxiv.org/abs/2104.04473 |
| Reducing Activation Recomputation in Large Transformer Models (sequence parallelism) | 2022 | https://arxiv.org/abs/2205.05198 |
| Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B | 2022 | https://arxiv.org/abs/2201.11990 |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | 2023 | https://arxiv.org/abs/2304.11277 |
| TorchTitan: One-stop PyTorch Native Solution for Production-Ready LLM Pre-training | 2024 | https://arxiv.org/abs/2410.06511 |

## Cross-references in AIForge

- [GPU Infrastructure and Scheduling](../GPU_Infrastructure_and_Scheduling/) — clusters, NCCL topology, Slurm/K8s scheduling that distributed jobs run on.
- [Model Optimization](../Model_Optimization/) — mixed precision, quantization, and activation checkpointing that complement parallelism.
- [Workflow Orchestration](../Workflow_Orchestration/) — pipelines that launch and manage multi-node training runs.
- [Cost Optimization and FinOps](../Cost_Optimization_and_FinOps/) — controlling the GPU-hour cost of large-scale training.

## Sources

- https://github.com/NVIDIA/Megatron-LM
- https://github.com/microsoft/DeepSpeed
- https://docs.pytorch.org/docs/stable/fsdp.html
- https://github.com/pytorch/torchtitan
- https://github.com/huggingface/accelerate
- https://docs.ray.io/en/latest/train/train.html
- https://github.com/horovod/horovod
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- https://github.com/kubeflow/training-operator
- https://github.com/volcano-sh/volcano
- https://arxiv.org/abs/1909.08053
- https://arxiv.org/abs/1910.02054
- https://arxiv.org/abs/1811.06965
- https://arxiv.org/abs/2104.04473
- https://arxiv.org/abs/2205.05198
- https://arxiv.org/abs/2201.11990
- https://arxiv.org/abs/2304.11277
- https://arxiv.org/abs/2410.06511
- https://mlcommons.org/benchmarks/training/

_Expanded from the seed gap-sweep entry. Contributions welcome (see CONTRIBUTING.md)._
