# AI Evaluation and Benchmarks

Resources for **LLM evaluation**, **benchmark design**, **LLM-as-a-judge**, **agent evals**, and **frontier capability assessments**.

## Frontier Benchmarks (2024-2026)

| Benchmark | Domain | Link |
|---|---|---|
| **MMLU / MMLU-Pro** | General knowledge | https://huggingface.co/datasets/cais/mmlu |
| **GPQA Diamond** | Hard graduate-level science | https://github.com/idavidrein/gpqa |
| **HLE (Humanity's Last Exam)** | Extremely hard expert questions | https://lastexam.ai/ |
| **MATH / AIME / IMO** | Competition mathematics | https://github.com/hendrycks/math |
| **SWE-bench Verified** | Real-world software engineering | https://www.swebench.com/ |
| **TerminalBench / TerminalBench 2.0** | CLI + system administration | https://www.tbench.ai/ |
| **ARC-AGI / ARC-AGI-2** | Abstract reasoning, novel skills | https://arcprize.org/ |
| **BIG-Bench / BIG-Bench Hard** | Diverse difficult tasks | https://github.com/google/BIG-bench |
| **LiveCodeBench** | Fresh competitive coding | https://livecodebench.github.io/ |
| **HumanEval / MBPP / EvalPlus** | Function-level code | https://github.com/openai/human-eval |
| **MMMU / MMMU-Pro** | Multimodal college-level | https://mmmu-benchmark.github.io/ |
| **VBench** | Video generation eval | https://vchitect.github.io/VBench-project/ |
| **OSWorld / WebArena / VisualWebArena** | Computer/web agent | https://os-world.github.io/ |
| **TAU-Bench** | Tool-augmented conversational | https://github.com/sierra-research/tau-bench |
| **MTEB / MMTEB** | Embedding model benchmark | https://huggingface.co/blog/mteb |
| **Chatbot Arena (LMSYS)** | Human pairwise preference | https://lmarena.ai/ |
| **ASR — LibriSpeech / FLEURS / Common Voice** | Speech recognition | https://commonvoice.mozilla.org/ |

## Safety / Robustness Benchmarks

- **HarmBench** — https://github.com/centerforaisafety/HarmBench
- **AdvBench** — https://github.com/llm-attacks/llm-attacks
- **WMDP (Weapons of Mass Destruction Proxy)** — https://www.wmdp.ai/
- **TruthfulQA** — https://github.com/sylinrl/TruthfulQA
- **ToxicChat** — https://huggingface.co/datasets/lmsys/toxic-chat
- **JailbreakBench** — https://jailbreakbench.github.io/

## LLM-as-a-Judge

- **MT-Bench** — https://github.com/lm-sys/FastChat
- **AlpacaEval 2.0** — https://github.com/tatsu-lab/alpaca_eval
- **G-Eval** — https://arxiv.org/abs/2303.16634
- **Prometheus** — https://huggingface.co/prometheus-eval

## Eval Frameworks

- **lm-evaluation-harness (EleutherAI)** — https://github.com/EleutherAI/lm-evaluation-harness
- **OpenAI Evals** — https://github.com/openai/evals
- **DeepEval** — https://github.com/confident-ai/deepeval
- **Inspect (UK AISI)** — https://inspect.ai-safety-institute.org.uk/
- **PromptFoo** — https://www.promptfoo.dev/
- **LangSmith Evaluations** — https://docs.smith.langchain.com/evaluation
- **Phoenix Evals (Arize)** — https://docs.arize.com/phoenix
- **Weave (W&B)** — https://wandb.ai/site/weave

## Leaderboards

- **Open LLM Leaderboard** — https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **LMSys Chatbot Arena** — https://lmarena.ai/leaderboard
- **Big Code Models Leaderboard** — https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- **Open VLM Leaderboard** — https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- **Aider LLM Coding Leaderboard** — https://aider.chat/docs/leaderboards/
- **SEAL (Scale Eval)** — https://scale.com/leaderboard

## Methodology

- **Evaluating LLMs is a Minefield** — Bowman 2023 — https://arxiv.org/abs/2310.18018
- **Are LLM Benchmarks Honest?** — https://arxiv.org/abs/2402.04580
- **Holistic Evaluation of Language Models (HELM)** — Liang et al. — https://crfm.stanford.edu/helm/
