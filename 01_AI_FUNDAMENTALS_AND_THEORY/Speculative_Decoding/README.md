# Speculative Decoding

> A lossless LLM inference-acceleration technique: a cheap "draft" mechanism proposes several future tokens that the target model verifies in a single parallel forward pass, accepting the longest valid prefix — yielding 2-4x speedups with **identical** output distribution.

## Why it matters

Autoregressive decoding is memory-bandwidth bound: each token needs a full forward pass that under-utilizes GPU compute. Speculative decoding turns that idle compute into useful work by verifying many candidate tokens at once, so a single target-model pass can emit several accepted tokens. Because verification uses rejection sampling (Leviathan et al.) or a typical-acceptance scheme, the output is provably equivalent to standard sampling — no quality loss. It is now a default acceleration path in vLLM, TGI, TensorRT-LLM, and SGLang.

## Taxonomy

| Approach | Idea | Draft source | Representative methods |
|---|---|---|---|
| **Independent draft model** | Small same-family LLM drafts, big LLM verifies | Separate small model | Leviathan et al., DeepMind speculative sampling |
| **Self-speculative** | One model drafts via early-exit / skipped layers | Subset of target's own layers | Draft & Verify, LayerSkip, Kangaroo, CLaSp |
| **Multi-head (extra heads)** | Lightweight heads predict multiple future tokens | Added decoding heads on target | Medusa |
| **Feature-level autoregression** | Draft at second-to-top hidden-feature level | Tiny autoregressive head | EAGLE, EAGLE-2, EAGLE-3 |
| **Tree / parallel verification** | Verify a token *tree* (many branches) at once | Any drafter | SpecInfer, EAGLE-2 dynamic trees, Sequoia |
| **Retrieval / n-gram (model-free)** | Copy candidate continuations from prompt/corpus | No weights | Prompt Lookup, vLLM n-gram, REST |
| **Self-drafting via Jacobi** | Parallel multi-token generation, no draft model | The model itself | Lookahead Decoding |
| **Online / adaptive** | Continuously adapt the drafter to live traffic | Updated draft model | Online Speculative Decoding |

## Key methods

| Method | One-liner | Speedup (reported) | Link |
|---|---|---|---|
| Speculative Decoding (Leviathan et al.) | Original draft-and-verify with rejection sampling | 2-3x | https://arxiv.org/abs/2211.17192 |
| Speculative Sampling (DeepMind) | Concurrent formulation, Chinchilla-70B | 2-2.5x | https://arxiv.org/abs/2302.01318 |
| SpecInfer | Tree-based parallel speculative verification | 1.5-2.8x | https://arxiv.org/abs/2305.09781 |
| Medusa | Extra decoding heads + tree attention | 2.2-3.6x | https://arxiv.org/abs/2401.10774 |
| EAGLE | Feature-level (second-to-top) autoregression | ~3x | https://arxiv.org/abs/2401.15077 |
| EAGLE-2 | Context-aware **dynamic** draft trees | 3.05-4.26x | https://arxiv.org/abs/2406.16858 |
| EAGLE-3 | Training-time test + multi-layer features | up to ~6.5x | https://arxiv.org/abs/2503.01840 |
| Lookahead Decoding | Jacobi parallel decoding, no draft model | 1.5-2.3x | https://arxiv.org/abs/2402.02057 |
| Self-Speculative (Draft & Verify) | Skip layers to draft from same model | ~1.99x | https://arxiv.org/abs/2309.08168 |
| LayerSkip | Early-exit + self-speculative decoding | up to ~2.16x | https://arxiv.org/abs/2404.16710 |
| Online Speculative Decoding | Adapt drafter to query distribution online | up to ~3.1x | https://arxiv.org/abs/2310.07177 |

## Key frameworks & tools

| Tool | Speculative methods supported | Link |
|---|---|---|
| vLLM | Draft model, EAGLE/EAGLE-3, Medusa, n-gram / prompt-lookup | https://docs.vllm.ai/en/latest/features/spec_decode.html |
| vLLM n-gram | Model-free prompt n-gram matching | https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ |
| Hugging Face TGI | Medusa + n-gram speculation | https://huggingface.co/docs/text-generation-inference/conceptual/speculation |
| HF Transformers | Assisted generation (`assistant_model`), prompt lookup | https://huggingface.co/blog/assisted-generation |
| TensorRT-LLM | Draft-target, Medusa, EAGLE, ReDrafter, lookahead | https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html |
| Medusa (reference impl) | Multiple decoding heads + tree attention | https://github.com/FasterDecoding/Medusa |
| EAGLE (reference impl) | EAGLE-1/2/3 feature-level drafting | https://github.com/SafeAILab/EAGLE |
| Lookahead Decoding | Jacobi self-drafting | https://github.com/hao-ai-lab/LookaheadDecoding |
| LayerSkip | Self-speculative early-exit | https://github.com/facebookresearch/LayerSkip |

## Benchmarks

| Benchmark | Use | Link |
|---|---|---|
| Spec-Bench | Standardized comparison of speculative-decoding methods across tasks | https://github.com/hemingkx/Spec-Bench |
| MT-Bench | Multi-turn quality/throughput target used by EAGLE/Medusa | https://huggingface.co/datasets/lmsys/mt_bench_human_judgments |
| HumanEval | Code task where acceptance rates are high (good for spec-dec) | https://github.com/openai/human-eval |
| CNN/DailyMail | Summarization where n-gram lookup excels (high copy overlap) | https://huggingface.co/datasets/abisee/cnn_dailymail |

Key metric: **mean accepted length** (tokens accepted per target pass) and **acceptance rate**, which together determine wall-clock speedup; both are context- and task-dependent.

## Key papers

- Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding* (ICML 2023) — https://arxiv.org/abs/2211.17192
- Chen et al. (DeepMind) — *Accelerating Large Language Model Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318
- Miao et al. — *SpecInfer: Tree-based Speculative Inference and Verification* (ASPLOS 2024) — https://arxiv.org/abs/2305.09781
- Cai et al. — *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* — https://arxiv.org/abs/2401.10774
- Li et al. — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — https://arxiv.org/abs/2401.15077
- Li et al. — *EAGLE-2: Faster Inference with Dynamic Draft Trees* — https://arxiv.org/abs/2406.16858
- Li et al. — *EAGLE-3: Scaling Inference Acceleration via Training-Time Test* — https://arxiv.org/abs/2503.01840
- Fu et al. — *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (ICML 2024) — https://arxiv.org/abs/2402.02057
- Zhang et al. — *Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding* (ACL 2024) — https://arxiv.org/abs/2309.08168
- Elhoushi et al. — *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* (ACL 2024) — https://arxiv.org/abs/2404.16710
- Xia et al. — *Unlocking Efficiency in LLM Inference: A Survey of Speculative Decoding* (ACL Findings 2024) — https://arxiv.org/abs/2401.07851

## Cross-references in AIForge

- [`../Long_Context_Models/`](../Long_Context_Models/) — KV-cache pressure and inference-time efficiency
- [`../Test_Time_Compute/`](../Test_Time_Compute/) — trading compute for quality/latency at inference
- [`../State_Space_Models/`](../State_Space_Models/) — alternative architectures targeting decode-time efficiency
- [`../Deep_Learning/`](../Deep_Learning/) — Transformer decoding fundamentals underpinning these methods

## Sources

- https://arxiv.org/abs/2211.17192
- https://arxiv.org/abs/2302.01318
- https://arxiv.org/abs/2305.09781
- https://arxiv.org/abs/2401.10774
- https://arxiv.org/abs/2401.15077
- https://arxiv.org/abs/2406.16858
- https://arxiv.org/abs/2503.01840
- https://arxiv.org/abs/2402.02057
- https://arxiv.org/abs/2309.08168
- https://arxiv.org/abs/2404.16710
- https://arxiv.org/abs/2401.07851
- https://docs.vllm.ai/en/latest/features/spec_decode.html
- https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/
- https://huggingface.co/docs/text-generation-inference/conceptual/speculation
- https://huggingface.co/blog/assisted-generation
- https://github.com/FasterDecoding/Medusa
- https://github.com/SafeAILab/EAGLE
- https://github.com/hemingkx/Spec-Bench

_Expanded from a verified high-value gap-sweep seed. Contributions welcome (see CONTRIBUTING.md)._
