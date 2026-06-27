# GPU Infrastructure and Scheduling

> The hardware substrate of production AI: partitioning, sharing, and scheduling GPUs across clusters — MIG/time-slicing, fractional GPUs, device plugins, gang scheduling, GPU operators, and high-speed fabric (NVLink/InfiniBand).

## Why it matters

GPUs are the most expensive line item in any AI platform, yet idle and fragmented GPUs are the norm: a single training or inference pod often pins a whole accelerator while using a fraction of it. GPU infrastructure (operators, device plugins, DCGM telemetry) and GPU-aware scheduling (gang/topology-aware placement, fractional sharing, queueing) reclaim that waste, enforce multi-tenant isolation, and make distributed training/serving actually schedulable. As of late 2025, Kubernetes Dynamic Resource Allocation (DRA) went GA, shifting fine-grained accelerator allocation into the core API.

## Taxonomy

| Sub-area | What it solves | Representative tech |
|---|---|---|
| **GPU partitioning** | Split one physical GPU into isolated/concurrent slices | MIG, time-slicing, MPS, vGPU |
| **Fractional / shared GPU** | Sub-GPU requests for notebooks & small inference | KAI Scheduler, Run:ai, device-plugin sharing |
| **Device plugins & operators** | Expose GPUs to the cluster, install drivers, telemetry | NVIDIA GPU Operator, k8s-device-plugin, DCGM |
| **Gang / batch scheduling** | All-or-nothing placement for distributed jobs | Volcano, Kueue, KAI, YuniKorn |
| **Topology-aware placement** | Co-locate pods on NVLink/IB to cut comm overhead | KAI, HiveD, network-sensitive schedulers |
| **Dynamic Resource Allocation** | Kubernetes-native, vendor-defined device requests | DRA (k8s 1.34 GA), ResourceClaims/ComputeDomains |
| **HPC/Slurm integration** | Bridge batch HPC schedulers with K8s | Slinky (Slurm-on-K8s), KubeRay |
| **Interconnect / fabric** | High-bandwidth GPU-to-GPU communication | NVLink/NVSwitch, InfiniBand, NCCL |

## Sharing modes compared

| Mode | Isolation | Hardware support | Best for |
|---|---|---|---|
| **MIG** (Multi-Instance GPU) | Full compute + memory isolation | A100/H100/H200/B100-class | Multi-tenant production |
| **Time-slicing** | None (round-robin time share) | Any CUDA GPU | Dev clusters, bursty/idle GPUs |
| **MPS** (Multi-Process Service) | Soft (spatial, shared address space) | Volta+ | Co-operative same-tenant inference |
| **vGPU** | Hypervisor-level | Licensed NVIDIA vGPU | VM-based multi-tenancy |

## Key tools & frameworks

| Tool | Role | Link |
|---|---|---|
| NVIDIA GPU Operator | Drivers, device plugin, DCGM, MIG manager on K8s | https://github.com/NVIDIA/gpu-operator |
| NVIDIA k8s-device-plugin | Expose GPUs; MIG / time-slicing config | https://github.com/NVIDIA/k8s-device-plugin |
| NVIDIA DCGM | GPU telemetry, health & utilization | https://github.com/NVIDIA/DCGM |
| Volcano | Gang scheduling for batch/HPC/AI on K8s (CNCF) | https://github.com/volcano-sh/volcano |
| Kueue | Kubernetes-native job queueing, quota & fair sharing | https://github.com/kubernetes-sigs/kueue |
| KAI Scheduler | Fractional GPU, gang & topology-aware AI scheduler | https://github.com/kai-scheduler/KAI-Scheduler |
| Apache YuniKorn | Resource scheduler with gang scheduling & queues | https://github.com/apache/yunikorn-core |
| HiveD | Topology-guaranteed scheduling for multi-tenant GPU pools | https://github.com/microsoft/hivedscheduler |
| Run:ai genv | GPU environment / fractional GPU management | https://github.com/run-ai/genv |
| Slinky (slurm-operator) | Run Slurm natively on Kubernetes (SchedMD/NVIDIA) | https://github.com/SlinkyProject/slurm-operator |
| KubeRay | Ray clusters/jobs on K8s with GPU scheduling | https://github.com/ray-project/kuberay |
| NCCL | Topology-aware GPU collective communication | https://github.com/NVIDIA/nccl |

## Standards & specs

| Spec | Description | Link |
|---|---|---|
| Kubernetes DRA | Dynamic Resource Allocation (GA in 1.34) | https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/ |
| NVIDIA MIG User Guide | Partitioning Ampere+/Hopper GPUs into instances | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ |
| GPU time-slicing (Operator docs) | Oversubscribe GPUs via time-slicing config | https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html |

## Key papers

| Paper | Venue / Year | Link |
|---|---|---|
| Gandiva: Introspective Cluster Scheduling for Deep Learning | OSDI 2018 | https://www.usenix.org/conference/osdi18/presentation/xiao |
| Tiresias: A GPU Cluster Manager for Distributed Deep Learning | NSDI 2019 | https://www.usenix.org/conference/nsdi19/presentation/gu |
| Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning | OSDI 2021 | https://arxiv.org/abs/2008.12260 |
| GPU Cluster Scheduling for Network-Sensitive Deep Learning | 2024 | https://arxiv.org/abs/2401.16492 |
| Leveraging Multi-Instance GPUs through moldable task scheduling | 2025 | https://arxiv.org/abs/2507.13601 |
| On the Partitioning of GPU Power among Multi-Instances | 2025 | https://arxiv.org/abs/2501.17752 |
| Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols | 2025 | https://arxiv.org/abs/2507.04786 |
| BandPilot: Performance- and Contention-Aware GPU Dispatching in AI Clusters | 2025 | https://arxiv.org/abs/2506.15595 |

## Cross-references in AIForge

- [Distributed Training](../Distributed_Training/) — the workloads gang scheduling and topology-aware placement exist to serve.
- [Inference Optimization](../Inference_Optimization/) — fractional GPUs and MPS/time-slicing for dense serving.
- [Cost Optimization and FinOps](../Cost_Optimization_and_FinOps/) — GPU utilization is the dominant AI cost lever.
- [Cloud Platforms](../Cloud_Platforms/) — managed GPU node pools and accelerator scheduling.

## Sources

- https://github.com/NVIDIA/gpu-operator
- https://github.com/kai-scheduler/KAI-Scheduler
- https://github.com/kubernetes-sigs/kueue
- https://github.com/volcano-sh/volcano
- https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
- https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- https://developer.nvidia.com/blog/running-large-scale-gpu-workloads-on-kubernetes-with-slurm/
- https://arxiv.org/abs/2401.16492
- https://www.usenix.org/conference/osdi18/presentation/xiao
- https://arxiv.org/abs/2008.12260

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
