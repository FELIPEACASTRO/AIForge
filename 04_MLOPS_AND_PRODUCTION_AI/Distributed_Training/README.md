# Distributed Training

> There is no dedicated section on multi-GPU/multi-node training, which is foundational production MLOps for any team training or fine-tuning large models. DeepSpeed exists only as a single file buried under Cloud_Platforms/Azure_Microsoft/deepspeed; FSDP, Megatron-LM, 3D parallelism (tensor/pipeline/data), Slurm, NCCL, and training operators are entirely absent (grep for FSDP/Slurm/distributed training returned 0 dedicated hits). This is a major hole for the 'most complete' goal.

## Key resources

| Resource | Link |
|---|---|
| NVIDIA Megatron-LM (tensor/pipeline/sequence parallelism) | https://github.com/NVIDIA/Megatron-LM |
| Microsoft DeepSpeed (ZeRO, ZeRO-Infinity) | https://github.com/microsoft/DeepSpeed |
| PyTorch FSDP (Fully Sharded Data Parallel) docs | https://pytorch.org/docs/stable/fsdp.html |
| NVIDIA NCCL (collective comms) user guide | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html |
| Kubeflow Training Operator (PyTorchJob/MPIJob) | https://github.com/kubeflow/training-operator |
| Volcano (batch/gang scheduling for distributed training on K8s) | https://github.com/volcano-sh/volcano |

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
