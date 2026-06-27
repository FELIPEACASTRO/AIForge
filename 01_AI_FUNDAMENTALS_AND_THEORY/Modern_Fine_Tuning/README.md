# Modern Fine-Tuning

Curated reference for **post-training** techniques: SFT, RLHF, DPO, GRPO, ORPO, LoRA/QLoRA, and parameter-efficient fine-tuning (PEFT).

## Supervised Fine-Tuning (SFT)

- **Instruction Tuning** — FLAN, T0, Self-Instruct
- **SFTTrainer (TRL)** — https://huggingface.co/docs/trl/sft_trainer
- **Axolotl** — https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** — https://github.com/hiyouga/LLaMA-Factory
- **Unsloth** (2-5x faster fine-tuning) — https://github.com/unslothai/unsloth
- **TorchTune** — https://github.com/pytorch/torchtune

## Preference Optimization

| Method | Paper | Notes |
|---|---|---|
| **RLHF / PPO** | InstructGPT (2022) — https://arxiv.org/abs/2203.02155 | Original recipe; complex |
| **DPO** | Rafailov 2023 — https://arxiv.org/abs/2305.18290 | No reward model needed |
| **IPO** | Azar 2023 — https://arxiv.org/abs/2310.12036 | Identity-PO, less overfitting |
| **KTO** | Ethayarajh 2024 — https://arxiv.org/abs/2402.01306 | Kahneman-Tversky single-side preferences |
| **ORPO** | Hong 2024 — https://arxiv.org/abs/2403.07691 | Reference-free, SFT+PO in one stage |
| **SimPO** | Meng 2024 — https://arxiv.org/abs/2405.14734 | Length-normalized reward |
| **GRPO** | DeepSeek-Math 2024 — https://arxiv.org/abs/2402.03300 | Group-relative; basis of R1 |
| **RLVR** | Reasoning models — verifiable rewards (math, code) |

## PEFT (Parameter-Efficient Fine-Tuning)

- **LoRA** — Hu et al. 2021 — https://arxiv.org/abs/2106.09685
- **QLoRA** — Dettmers et al. 2023 — https://arxiv.org/abs/2305.14314
- **DoRA** — Liu et al. 2024 — https://arxiv.org/abs/2402.09353
- **LoRA+** — Hayou et al. 2024 — https://arxiv.org/abs/2402.12354
- **PiSSA** — Meng 2024 — https://arxiv.org/abs/2404.02948
- **rsLoRA / VeRA / LongLoRA / GaLore**
- **HF PEFT library** — https://huggingface.co/docs/peft

## Distillation

- **Knowledge Distillation** — Hinton 2015 — https://arxiv.org/abs/1503.02531
- **Distilling Step-by-Step** — Hsieh 2023 — https://arxiv.org/abs/2305.02301
- **Sequence-Level KD** — Kim & Rush 2016
- **MiniLLM** — https://arxiv.org/abs/2306.08543
- **Distill-Whisper** — https://github.com/huggingface/distil-whisper

## Reasoning Training (post-2024)

- **DeepSeek-R1** (GRPO + RL on verifiable rewards) — https://arxiv.org/abs/2501.12948
- **Open-R1 (HF)** — https://github.com/huggingface/open-r1
- **TÜLU 3** (post-training recipe by AI2) — https://arxiv.org/abs/2411.15124
- **Self-Rewarding LMs** — Yuan 2024 — https://arxiv.org/abs/2401.10020
- **Process Reward Models (PRM)** — Lightman 2023 — https://arxiv.org/abs/2305.20050

## Datasets for Post-Training

- **UltraChat / UltraFeedback** — HuggingFaceH4
- **OpenOrca** — https://huggingface.co/datasets/Open-Orca/OpenOrca
- **Tulu-3-SFT-Mixture** — https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- **HelpSteer2 (NVIDIA)** — https://huggingface.co/datasets/nvidia/HelpSteer2
- **OpenMathInstruct-2** — https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

## Compute Hardware Notes

- Single A100/H100 + QLoRA fine-tunes 7-13B models in hours
- Full fine-tunes of 70B require 8x H100 + DeepSpeed/FSDP
- **DeepSpeed ZeRO-3** + **FSDP** — multi-GPU sharded training
- **Liger Kernels** — fused Triton kernels — https://github.com/linkedin/Liger-Kernel
