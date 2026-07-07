# Benchmark And Evaluation Source Index - 2026-07-07

This file organizes benchmark and evaluation sources for model, agent, prompt, dataset, and production AI assessment. The goal is to avoid treating leaderboard numbers as truth without checking task definition, split, contamination risk, evaluator implementation, and update date.

## General Evaluation Frameworks

| Source | Scope | Evidence to record |
|---|---|---|
| [MLPerf benchmarks](https://mlcommons.org/benchmarks/) | Training, inference, client, HPC, storage, and broader ML system performance. | Version, division, hardware, software stack, accuracy target, latency, throughput, and power notes. |
| [HELM latest](https://crfm.stanford.edu/helm/latest/) | Holistic evaluation of language models with transparent prompts and scenarios. | Scenario, metric, model version, prompt transparency, and raw-completion availability. |
| [HELM GitHub](https://github.com/stanford-crfm/helm) | Open-source framework for reproducible HELM evaluations. | Commit, scenario config, run command, and environment. |
| [HELM capabilities](https://crfm.stanford.edu/helm/capabilities/latest/) | Curated capability leaderboard for language models. | Capability, scenario, prompt set, leaderboard date, and reproducibility path. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Unified LLM benchmark runner with many academic tasks. | Harness version, task config, model backend, tokenizer, few-shot setting, and generation parameters. |
| [OpenAI Evals](https://github.com/openai/evals) | Eval authoring and model behavior testing. | Eval spec, samples, grader, model-as-judge prompt, and pass criteria. |
| [Promptfoo](https://www.promptfoo.dev/) | Prompt, provider, RAG, and red-team evals. | Test matrix, providers, assertions, dataset, red-team plugins, and CI status. |

## LLM, Coding, And Agent Benchmarks

| Source | Scope | Caveat |
|---|---|---|
| [SWE-bench leaderboards](https://www.swebench.com/) | Official leaderboards for repository-level software issue resolution. | Compare scaffold, model, cost, step limit, and benchmark variant. |
| [SWE-bench overview](https://www.swebench.com/SWE-bench/) | Original benchmark definition and ecosystem. | Use task definition before citing leaderboard scores. |
| [SWE-bench Verified](https://www.swebench.com/verified.html) | Human-filtered 500-instance subset created with OpenAI collaboration. | Human validation improves quality but does not remove saturation or scaffold effects. |
| [SWE-bench GitHub](https://github.com/swe-bench/SWE-bench) | Evaluation code, dataset access, and Dockerized execution. | Cite exact commit and Docker environment. |
| [SWE-bench Hugging Face dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | Dataset mirror for the original benchmark. | Confirm split and benchmark variant. |
| [GPQA paper](https://arxiv.org/abs/2311.12022) | Graduate-level Google-proof STEM QA benchmark. | Track variant such as full, diamond, or filtered subsets. |
| [GPQA GitHub](https://github.com/idavidrein/gpqa) | Data and baseline repository for GPQA. | Check canary and contamination guidance. |
| [Chatbot Arena leaderboard](https://huggingface.co/spaces/lmarena-ai/arena-leaderboard) | Human preference arena across model families and modalities. | Arena scores depend on user population, prompts, and active model pool. |
| [BIG-bench](https://github.com/google/BIG-bench) | Large collection of challenging language-model tasks. | Many tasks are older; check saturation and contamination. |
| [HumanEval](https://github.com/openai/human-eval) | Python code-generation benchmark. | Narrow function synthesis; not a proxy for full software engineering. |
| [MATH](https://github.com/hendrycks/math) | Competition-style math benchmark. | Check exact subset and prompting format. |

## Dataset And ML Benchmark Discovery

| Source | Scope | AIForge use |
|---|---|---|
| [Papers with Code datasets](https://paperswithcode.com/datasets) | Dataset and benchmark discovery linked to papers and tasks. | Good for finding tasks, but verify upstream source/license. |
| [Papers with Code methods](https://paperswithcode.com/methods) | Method/task mapping and leaderboards. | Useful for routing models to topic directories. |
| [OpenML benchmark suites](https://www.openml.org/search?type=benchmark) | Structured ML benchmark suites and reproducible tasks. | Strong source for classical ML and tabular benchmarks. |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Classic ML datasets with metadata and download access. | Use for educational and baseline datasets. |
| [Hugging Face Datasets](https://huggingface.co/datasets) | Community and official datasets with viewers and metadata. | Verify license, split, and dataset card. |

## Required Benchmark Metadata

- Benchmark name, version, task definition, dataset source, split, license, and last checked date.
- Evaluator implementation, command, commit hash, environment, model/scaffold, prompt, and generation settings.
- Metric definition, confidence interval or uncertainty, cost, latency, failure rate, and manual-review policy.
- Contamination risk, saturation status, known gaming paths, and whether the benchmark matches a real product requirement.
