# GPU Infrastructure and Scheduling

> GPU cluster infrastructure — MIG/time-slicing, GPU sharing/fractional GPUs, device plugins, gang scheduling, GPU operator, fabric (NVLink/InfiniBand) — has no dedicated section. This is the hardware substrate of all production AI and is unrepresented (grep 'KEDA' 0, no GPU operator/device-plugin coverage).

## Key resources

| Resource | Link |
|---|---|
| NVIDIA GPU Operator (drivers, device plugin, DCGM on K8s) | https://github.com/NVIDIA/gpu-operator |
| NVIDIA k8s-device-plugin (MIG / time-slicing) | https://github.com/NVIDIA/k8s-device-plugin |
| Run:ai genv (GPU environment management) | https://github.com/run-ai/genv |
| NVIDIA DCGM (GPU telemetry/health) | https://github.com/NVIDIA/DCGM |
| Volcano (gang scheduling for GPU jobs) | https://github.com/volcano-sh/volcano |

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
