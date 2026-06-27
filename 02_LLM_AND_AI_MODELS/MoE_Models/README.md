# Mixture-of-Experts (MoE) Models

Sparse expert models that scale capacity without proportional compute cost.

## Open MoE Models

| Model | Total / Active | Org | Link |
|---|---|---|---|
| **Mixtral 8x7B / 8x22B** | 47B/13B, 141B/39B | Mistral | https://huggingface.co/mistralai |
| **DeepSeek-V3 / V3.2-Exp** | 671B / 37B | DeepSeek | https://github.com/deepseek-ai/DeepSeek-V3 |
| **DeepSeek-R1** | 671B / 37B | DeepSeek | https://github.com/deepseek-ai/DeepSeek-R1 |
| **Qwen2.5-MoE / Qwen3-MoE** | varies | Alibaba | https://github.com/QwenLM |
| **Grok-1** | 314B / 86B | xAI | https://github.com/xai-org/grok-1 |
| **GLM-4.5 / GLM-4.6** | MoE variants | Zhipu | https://github.com/THUDM/GLM-4 |
| **Kimi K2** | MoE | Moonshot | https://github.com/MoonshotAI/Kimi-K2 |
| **MiniMax-Text-01** | 456B / 45.9B | MiniMax | https://github.com/MiniMax-AI/MiniMax-01 |
| **Snowflake Arctic** | 480B / 17B | Snowflake | https://huggingface.co/Snowflake/snowflake-arctic-instruct |
| **DBRX** | 132B / 36B | Databricks | https://huggingface.co/databricks/dbrx-base |
| **Jamba** (SSM-MoE hybrid) | 52B / 12B | AI21 | https://huggingface.co/ai21labs/Jamba-v0.1 |
| **OLMoE** | 7B / 1B | AI2 | https://allenai.org/olmoe |
| **Phi-3.5-MoE** | 42B / 6.6B | Microsoft | https://huggingface.co/microsoft/Phi-3.5-MoE-instruct |

## Key Papers

- **Outrageously Large Neural Networks (Sparsely-Gated MoE)** — Shazeer 2017 — https://arxiv.org/abs/1701.06538
- **Switch Transformers** — Fedus 2021 — https://arxiv.org/abs/2101.03961
- **GShard** — Lepikhin 2020 — https://arxiv.org/abs/2006.16668
- **Expert Choice Routing** — Zhou 2022 — https://arxiv.org/abs/2202.09368
- **ST-MoE** — Zoph 2022 — https://arxiv.org/abs/2202.08906
- **DeepSeek-V3 Technical Report** — https://arxiv.org/abs/2412.19437
- **MoE-LLaVA** — https://arxiv.org/abs/2401.15947

## Training Frameworks

- **Megablocks** — dropless MoE — https://github.com/databricks/megablocks
- **DeepSpeed-MoE** — https://www.deepspeed.ai/tutorials/mixture-of-experts/
- **Tutel (Microsoft)** — https://github.com/microsoft/tutel
- **OpenMoE** — https://github.com/XueFuzhao/OpenMoE
- **vLLM / SGLang / TensorRT-LLM** — all support MoE inference
