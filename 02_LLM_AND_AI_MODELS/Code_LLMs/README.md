# Code LLMs

> Code LLMs are large language models specialized for software tasks — code generation, completion, fill-in-the-middle (FIM), repair, explanation, and increasingly autonomous "agentic" engineering over whole repositories.

## Why it matters

Code is one of the highest-value and most measurable LLM verticals: it powers Copilot, Cursor, Windsurf, and agentic coding tools, and it drives a tight feedback loop where outputs are executable and unit-testable. Code-specialized models (StarCoder2, Code Llama, DeepSeek-Coder, Qwen2.5/3-Coder, CodeGemma, Codestral) routinely match or beat much larger general models on coding benchmarks at a fraction of the cost. The frontier has shifted from single-function generation (HumanEval) to repository-scale, multi-step agentic work (SWE-bench), making this a distinct, fast-moving area separate from general [Text_LLMs](../Text_LLMs/).

## Taxonomy

| Sub-area | What it covers | Representative work |
|---|---|---|
| Code completion / FIM | IDE autocomplete, infilling between prefix/suffix | Codestral, StarCoder2, CodeGemma |
| Instruction code generation | NL prompt → function/program | Qwen2.5-Coder-Instruct, Code Llama - Instruct |
| Agentic SWE | Repo-level issue resolution, tool/browser use | Qwen3-Coder, SWE-agent, OpenHands |
| Code reasoning / repair | Bug fixing, test-driven repair, self-debug | DeepSeek-Coder-V2, reasoning models |
| Open & reproducible | Fully open data + training recipes | StarCoder2 / The Stack v2, OpenCoder |
| Specialized small models | On-device / edge code assist | Qwen2.5-Coder-0.5B/1.5B, CodeGemma-2B |

## Key models

| Model family | Org | Sizes / notes | Link |
|---|---|---|---|
| Qwen3-Coder | Alibaba | MoE up to 480B-A35B; agentic SOTA among open models | https://qwenlm.github.io/blog/qwen3-coder/ |
| Qwen2.5-Coder | Alibaba | 0.5B–32B; 32B-Instruct rivals GPT-4o on code | https://github.com/QwenLM/Qwen2.5-Coder |
| DeepSeek-Coder-V2 | DeepSeek | MoE, 338 languages, 128K ctx | https://github.com/deepseek-ai/DeepSeek-Coder-V2 |
| DeepSeek-Coder (v1) | DeepSeek | 1.3B–33B, repo-level pretraining | https://github.com/deepseek-ai/DeepSeek-Coder |
| StarCoder2 | BigCode | 3B/7B/15B, fully open (The Stack v2) | https://github.com/bigcode-project/starcoder2 |
| Code Llama | Meta | 7B–70B, Python/Instruct variants | https://arxiv.org/abs/2308.12950 |
| Codestral | Mistral AI | 22B, FIM-optimized for IDE autocomplete | https://mistral.ai/news/codestral/ |
| CodeGemma | Google | 2B/7B, lightweight FIM + generation | https://huggingface.co/google/codegemma-7b |
| OpenCoder | INF / M-A-P | 1.5B/8B, fully open data + recipe | https://huggingface.co/papers/2411.04905 |

## Tools & frameworks (agentic coding)

| Tool | Purpose | Link |
|---|---|---|
| SWE-agent | Agent scaffold for resolving GitHub issues | https://github.com/SWE-agent/SWE-agent |
| OpenHands (ex-OpenDevin) | Open agent platform for software dev | https://github.com/All-Hands-AI/OpenHands |
| Aider | Terminal AI pair-programming over a git repo | https://github.com/Aider-AI/aider |
| Continue | Open-source IDE autocomplete/chat extension | https://github.com/continuedev/continue |
| Cline | Autonomous coding agent in VS Code | https://github.com/cline/cline |

## Benchmarks

| Benchmark | Focus | Link |
|---|---|---|
| HumanEval | 164 function-level Python problems (pass@k) | https://github.com/openai/human-eval |
| MBPP | ~1,000 entry-level Python problems | https://github.com/google-research/google-research/tree/master/mbpp |
| EvalPlus (HumanEval+/MBPP+) | 80× more tests; catches edge-case failures | https://github.com/evalplus/evalplus |
| BigCodeBench | 1,140 library-aware tasks, 139 libraries | https://github.com/bigcode-project/bigcodebench |
| LiveCodeBench | Contamination-free contest problems (live) | https://github.com/LiveCodeBench/LiveCodeBench |
| SWE-bench | Real GitHub issues, repo-level resolution | https://www.swebench.com/ |
| MultiPL-E | HumanEval/MBPP across 18+ languages | https://github.com/nuprl/MultiPL-E |
| CRUXEval | Code reasoning / input-output prediction | https://github.com/facebookresearch/cruxeval |

## Datasets

| Dataset | Description | Link |
|---|---|---|
| The Stack v2 | ~67TB permissively-licensed source code (StarCoder2) | https://huggingface.co/datasets/bigcode/the-stack-v2 |
| The Stack (v1) | 6TB de-duplicated source code, 358 languages | https://huggingface.co/datasets/bigcode/the-stack |
| CodeSearchNet | Code + NL docstring pairs, 6 languages | https://github.com/github/CodeSearchNet |
| CommitPackFT | Filtered git commits for instruction tuning | https://huggingface.co/datasets/bigcode/commitpackft |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Evaluating LLMs Trained on Code (Codex / HumanEval) | 2021 | https://arxiv.org/abs/2107.03374 |
| Code Llama: Open Foundation Models for Code | 2023 | https://arxiv.org/abs/2308.12950 |
| StarCoder 2 and The Stack v2 | 2024 | https://arxiv.org/abs/2402.19173 |
| DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models | 2024 | https://arxiv.org/abs/2406.11931 |
| Qwen2.5-Coder Technical Report | 2024 | https://arxiv.org/abs/2409.12186 |
| OpenCoder: Open Cookbook for Top-Tier Code LLMs | 2024 | https://arxiv.org/abs/2411.04905 |
| SWE-bench: Can LMs Resolve Real-World GitHub Issues? | 2023 | https://arxiv.org/abs/2310.06770 |
| BigCodeBench: Benchmarking Code Gen with Diverse Function Calls | 2024 | https://arxiv.org/abs/2406.15877 |
| LiveCodeBench: Holistic & Contamination-Free Evaluation | 2024 | https://arxiv.org/abs/2403.07974 |

## Cross-references in AIForge

- [Text_LLMs](../Text_LLMs/) — general-purpose foundation models that code LLMs branch from
- [Reasoning_Models](../Reasoning_Models/) — chain-of-thought / test-time compute, increasingly fused with agentic coding
- [MoE_Models](../MoE_Models/) — the architecture behind DeepSeek-Coder-V2 and Qwen3-Coder
- [Small_Language_Models](../Small_Language_Models/) — on-device code assist (CodeGemma-2B, Qwen2.5-Coder-0.5B/1.5B)

## Sources

- https://github.com/bigcode-project/starcoder2
- https://arxiv.org/abs/2402.19173
- https://arxiv.org/abs/2308.12950
- https://github.com/deepseek-ai/DeepSeek-Coder-V2
- https://arxiv.org/abs/2406.11931
- https://github.com/QwenLM/Qwen2.5-Coder
- https://arxiv.org/abs/2409.12186
- https://qwenlm.github.io/blog/qwen3-coder/
- https://mistral.ai/news/codestral/
- https://huggingface.co/google/codegemma-7b
- https://huggingface.co/papers/2411.04905
- https://github.com/bigcode-project/bigcodebench
- https://www.swebench.com/ , https://arxiv.org/abs/2310.06770
- https://arxiv.org/abs/2403.07974 , https://arxiv.org/abs/2406.15877
- https://arxiv.org/abs/2107.03374
- https://huggingface.co/datasets/bigcode/the-stack-v2

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
