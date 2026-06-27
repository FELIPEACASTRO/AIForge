# Cost Optimization and FinOps

> Practices, tools, and techniques for measuring, attributing, and reducing the cost of training and serving AI/ML workloads — spanning GPU utilization, inference efficiency, spot/preemptible scheduling, autoscaling-to-zero, token-cost tracking, and multi-cloud arbitrage.

## Why it matters

Compute is the dominant line item for production GenAI, and idle or under-utilized accelerators waste money silently — surveys repeatedly find a large share of provisioned GPU capacity goes unused. FinOps brings engineering, finance, and product together to make cost a first-class, observable metric (cost-per-token, cost-per-request, cost-per-feature) rather than a quarterly surprise. Because a model/runtime choice compounds across every subsequent inference call, optimizing the serving stack and matching infrastructure to the workload pattern can cut spend by an order of magnitude with minimal quality loss.

## Taxonomy

| Sub-area | What it covers | Representative levers |
|---|---|---|
| **Cost visibility & attribution** | Per-team/per-workload/per-token cost allocation, anomaly detection, unit economics | OpenCost, Kubecost, CloudZero, cloud cost explorers |
| **GPU utilization & telemetry** | Measuring real accelerator usage vs. allocation, right-sizing | DCGM exporter, Prometheus/Grafana, MIG partitioning |
| **Inference efficiency** | Higher throughput per GPU at fixed latency (batching, KV-cache, prefix reuse) | vLLM, SGLang, TensorRT-LLM, continuous batching |
| **Model compression** | Fewer FLOPs/bytes per token at acceptable accuracy | Quantization (FP8/INT4), pruning, distillation, speculative decoding |
| **Elastic scheduling** | Scale-to-zero, just-in-time provisioning, bin-packing | KEDA, Karpenter, Knative, Ray Serve autoscaling |
| **Capacity arbitrage** | Spot/preemptible instances, multi-cloud/region price shopping | SkyPilot, spot fleets, cross-cloud orchestration |
| **Routing & caching** | Send cheap requests to cheap models; cache repeated work | LLM gateways, semantic caching, model cascades |
| **Governance / FinOps practice** | Budgets, showback/chargeback, forecasting, culture | FinOps Foundation Framework, tagging policies |

## Key tools

### Cost visibility & FinOps platforms

| Tool | Scope | Link |
|---|---|---|
| OpenCost (CNCF) | Open-source Kubernetes + cloud cost allocation; GPU-aware | https://github.com/opencost/opencost |
| Kubecost | OpenCost-based with NVIDIA GPU utilization/efficiency cost monitoring | https://github.com/kubecost |
| CloudZero | Cost-per-unit (per inference/conversation/feature) + anomaly detection | https://www.cloudzero.com/ |
| FinOps Foundation Framework | Vendor-neutral FinOps capabilities, phases, and principles | https://www.finops.org/framework/ |
| OpenTelemetry / Prometheus + Grafana | Metrics backbone for cost & utilization dashboards | https://opentelemetry.io/ |

### GPU utilization & scheduling

| Tool | Role | Link |
|---|---|---|
| NVIDIA DCGM Exporter | GPU telemetry (`DCGM_FI_DEV_GPU_UTIL`) to Prometheus | https://github.com/NVIDIA/dcgm-exporter |
| Karpenter | Just-in-time node provisioning, spot-by-default, bin-packing (EKS) | https://github.com/aws/karpenter-provider-aws |
| KEDA | Event-driven autoscaling incl. scale-to-zero | https://github.com/kedacore/keda |
| Knative | Request-driven serverless scale-to-zero for model serving | https://github.com/knative/serving |
| Ray Serve | Queue-depth-based replica autoscaling for LLM serving | https://github.com/ray-project/ray |
| NVIDIA MIG / time-slicing | Partition one GPU across small workloads | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ |

### Multi-cloud & spot orchestration

| Tool | Role | Link |
|---|---|---|
| SkyPilot | Run/scale jobs across clouds, managed spot, cost arbitrage | https://github.com/skypilot-org/skypilot |
| dstack | Open-source orchestration for GPU dev/training/inference across providers | https://github.com/dstackai/dstack |
| Cluster Autoscaler | Node-level scale-down of idle capacity (K8s) | https://github.com/kubernetes/autoscaler |

### Inference efficiency engines

| Tool | Key technique | Link |
|---|---|---|
| vLLM | PagedAttention + continuous batching (2–4× throughput) | https://github.com/vllm-project/vllm |
| SGLang | RadixAttention prefix-cache reuse, structured generation | https://github.com/sgl-project/sglang |
| TensorRT-LLM | NVIDIA-optimized kernels, FP8/INT4, in-flight batching | https://github.com/NVIDIA/TensorRT-LLM |
| Hugging Face TGI | Production text-generation serving with tensor parallelism | https://github.com/huggingface/text-generation-inference |
| llama.cpp | Quantized CPU/edge inference (GGUF) for cheap deployment | https://github.com/ggml-org/llama.cpp |

## Key papers

| Year | Paper | Why it matters | Link |
|---|---|---|---|
| 2022 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | IO-aware tiling cuts attention memory traffic; foundational for cheap long-context | https://arxiv.org/abs/2205.14135 |
| 2022 | SmoothQuant: Accurate and Efficient PTQ for LLMs | Training-free W8A8 quantization preserving accuracy | https://arxiv.org/abs/2211.10438 |
| 2022 | Fast Inference from Transformers via Speculative Decoding | 2–3× faster decoding with identical outputs via draft model | https://arxiv.org/abs/2211.17192 |
| 2023 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | One-shot 3–4 bit weight quantization at scale | https://arxiv.org/abs/2210.17323 |
| 2023 | Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | Near-zero KV-cache waste; 2–4× serving throughput | https://arxiv.org/abs/2309.06180 |
| 2024 | AWQ: Activation-aware Weight Quantization for On-Device LLMs | Hardware-friendly low-bit weight-only quantization | https://arxiv.org/abs/2306.00978 |
| 2025 | Taming the Titans: A Survey of Efficient LLM Inference Serving | Broad survey of serving-cost optimization techniques | https://arxiv.org/abs/2504.19720 |

## Practical cost levers (rules of thumb)

- **Quantize first.** FP8 on H100 (native in vLLM/TensorRT-LLM) typically yields 1.3–2× throughput at <2% quality loss; INT4 weight-only (AWQ/GPTQ) shrinks memory so larger models fit on one GPU.
- **Batch continuously.** Continuous/in-flight batching (vLLM, TGI) keeps the GPU saturated instead of idling between requests.
- **Reuse prefixes.** RadixAttention/prefix caching (SGLang) avoids recomputing shared system prompts across requests.
- **Route by difficulty.** Cascade cheap models first; escalate to large models only when needed (see LLM Gateway and Routing).
- **Scale to zero.** KEDA/Knative for spiky traffic; Karpenter + spot for batch and fault-tolerant serving.
- **Measure unit economics.** Track cost-per-token and cost-per-request, not just monthly cloud totals.

## Cross-references in AIForge

- [Inference Optimization](../Inference_Optimization/) — batching, KV-cache, and serving-throughput techniques
- [Model Optimization](../Model_Optimization/) — quantization, pruning, and distillation methods
- [GPU Infrastructure and Scheduling](../GPU_Infrastructure_and_Scheduling/) — accelerator provisioning and sharing
- [LLM Gateway and Routing](../LLM_Gateway_and_Routing/) — model cascades, caching, and cost-aware routing
- [AI Observability](../AI_Observability/) — telemetry and metrics that feed cost dashboards

## Sources

- vLLM / PagedAttention paper — https://arxiv.org/abs/2309.06180
- FlashAttention — https://arxiv.org/abs/2205.14135
- Speculative decoding — https://arxiv.org/abs/2211.17192
- SmoothQuant — https://arxiv.org/abs/2211.10438 · GPTQ — https://arxiv.org/abs/2210.17323 · AWQ — https://arxiv.org/abs/2306.00978
- Survey of efficient LLM inference serving — https://arxiv.org/abs/2504.19720
- OpenCost — https://github.com/opencost/opencost · Kubecost GPU monitoring — https://www.apptio.com/blog/gpu-monitoring/
- NVIDIA DCGM Exporter — https://github.com/NVIDIA/dcgm-exporter
- SkyPilot — https://github.com/skypilot-org/skypilot · Karpenter — https://github.com/aws/karpenter-provider-aws · KEDA — https://github.com/kedacore/keda
- FinOps Foundation Framework — https://www.finops.org/framework/
- CloudZero inference cost guide — https://www.cloudzero.com/blog/inference-cost/

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
