# Agentic & LLM Trading Agents (Frontier 2024-2026)

> A dense, source-verified map of the **2023-2026 frontier** where large language models (LLMs) and multi-agent ("agentic") systems are applied to trading, investing, alpha mining, and equity research — with real GitHub/arXiv/Hugging Face links, an honest read on hype vs. evidence, and Brazil-access notes. Mainstream venues already indexed elsewhere in this repo (US NYSE/Nasdaq, B3, NSE/BSE, SSE/SZSE, generic data APIs, Kaggle/HF, arXiv q-fin) are referenced only where they connect to agents.

This page covers a research line that exploded after ChatGPT (late 2022) and is now (mid-2026) a crowded, fast-moving subfield. Treat it as a **reading list + radar**, not investment advice. A recurring, well-documented caveat runs through the whole area: backtested "alpha" from LLM agents is frequently inflated by **information leakage / pre-training contamination** — see the *Reality check* section. (Portuguese: agentes de negociação com IA / agentes autônomos de investimento.)

---

## 1. The hub: AI4Finance-Foundation

Most open-source momentum in this space orbits one GitHub organization. If you read nothing else, read this org.

| Project | What it is | Link |
|---|---|---|
| **AI4Finance-Foundation** (org) | Open-source financial AI hub: FinGPT, FinRL, FinRobot, FinNLP, datasets, tutorials. Community-driven, very active. | https://github.com/AI4Finance-Foundation · https://ai4finance.org/ |
| **FinGPT** | Open-source financial LLM stack (data-centric, LoRA-tuned). Positioned as the open alternative to BloombergGPT. Paper: arXiv:2306.06031. Models on HF. | https://github.com/AI4Finance-Foundation/FinGPT · https://huggingface.co/FinGPT · https://arxiv.org/abs/2306.06031 |
| **FinGPT-Forecaster** | LoRA fine-tune of Llama-2-7b-chat on ~1yr DOW30 news+financials; outputs "positive developments / potential concerns" + a next-week direction call. Released Nov 2023. | https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster |
| **FinRL** | First open framework for **deep reinforcement learning** in trading (not LLM, but the agent substrate many LLM systems plug into). Covers DJIA, S&P 500, NASDAQ-100, HSI, SSE 50, CSI 300; portfolio, crypto, HFT. Paper: arXiv:2011.09607. | https://github.com/AI4Finance-Foundation/FinRL |
| **FinRL-Meta** | Dynamic datasets + standardized market environments ("metaverse") for FinRL agents. | https://github.com/AI4Finance-Foundation/FinRL-Meta |
| **FinRobot** | Multi-agent **platform** layered on top of FinGPT: LLMs + RL + quant tools for equity research, forecasting, report generation. `pip install -U finrobot`. | https://github.com/AI4Finance-Foundation/FinRobot |

> Note: AI4Finance projects are research/education-grade. Stars and ambition are high; live-trading robustness is not guaranteed. Read the issues before trusting any backtest.

---

## 2. Multi-agent LLM trading frameworks (the headline systems)

These mimic a trading firm: specialized LLM agents (analysts, researchers, traders, risk) debate and decide. This is the most-cited cluster.

| Project | What it does | LLM-agent pattern | Link (GitHub / arXiv / HF) |
|---|---|---|---|
| **TradingAgents** (Tauric Research, UCLA/MIT) | Simulates a trading firm: fundamental/sentiment/news/technical analysts → bull vs. bear researcher **debate** → trader → risk team. Most popular recent repo in the space. | Role-specialized agents + structured debate + risk gate | https://github.com/TauricResearch/TradingAgents · arXiv:2412.20138 (Dec 2024) |
| **FinAgent** (a.k.a. *A Multimodal Foundation Agent for Financial Trading*) | Tool-augmented, multimodal (numeric+text+visual) trading agent with a **dual-level reflection** module + diversified memory. KDD 2024. Reports large profit gains vs. 9 baselines (treat with caution — single-asset backtests). | Multimodal market intelligence + dual reflection + memory retrieval | arXiv:2402.18485 |
| **FinMem** (Stevens Institute) | Performance-enhanced single-agent trader with **layered memory** (short/mid/long, decay) + configurable "character"/risk profile. AAAI Spring Symposium 2024. | Profiling + layered memory + decision module | arXiv:2311.13743 |
| **TradingGPT** | Multi-agent system with **layered memory + distinct characters** and an inter-agent **debate** mechanism over shared holdings. Conceptual precursor to FinMem/TradingAgents. | Layered memory + character design + peer debate | arXiv:2309.03736 |
| **FinCon** (Stevens/Columbia et al.) | Manager–analyst hierarchy with an **Actor-Critic** loop and *Conceptual Verbal Reinforcement (CVRF)* — episodic reflection on wins/losses + risk-control self-critique. NeurIPS 2024. | Hierarchical agents + verbal RL + risk-control belief updates | arXiv:2407.06567 |
| **HedgeAgents** (SCUT) | Central fund manager + asset-class "hedging experts" coordinating via three conference types; balanced/hedging-aware. WWW 2025 Companion. Reports 70% annualized / 400% total over 3yr (backtest — verify before believing). | Manager + specialist experts + scheduled "conferences" | arXiv:2502.13165 · https://hedgeagents.github.io |
| **StockAgent** (*When AI Meets Finance*) | Multi-agent **A-share-flavored simulation** of free trading with no prior market data; studies how external events move agent behavior/prices. Useful for ABM-style research, not a strategy. | Agent-based market simulation (init/trade/post-trade/event phases) | https://github.com/MingyuJ666/Stockagent · arXiv:2407.18957 |
| **TiMi (Trade in Minutes!)** | Decouples strategy *design* from minute-level *deployment* (no continuous inference at runtime). Uses DeepSeek-V3 (semantics), Qwen2.5-Coder-32B (code), DeepSeek-R1 (math). Live tests on 200+ pairs, US index futures + crypto, Jan–Apr 2025. ICLR 2026. | Tool-specialized agents + closed-loop math reflection + compiled bot | arXiv:2510.04787 |
| **StockSim** | Dual-mode, **order-level** simulator for evaluating multi-agent LLMs in markets (microstructure-aware backtesting). | Evaluation/simulation harness for agents | arXiv:2507.09255 |

**Brazil access note:** these agents are research code; to actually trade Brazilian assets you would wire them to B3 data (already covered in this repo) or trade US tickers/ETFs/BDRs. None ships a B3 connector out of the box — treat that as an integration task.

---

## 3. LLMs for alpha mining & factor discovery

A distinct, fast-growing branch: LLMs **generate trading signals / factor formulas / strategy code** rather than directly placing trades.

| Project | What it does | LLM-agent pattern | Link |
|---|---|---|---|
| **Alpha-GPT** (Shanghai/HKUST) | Human-AI **interactive** alpha mining: prompt-engineering pipeline turns a quant's natural-language idea into candidate alphas. | Human-in-the-loop prompt → alpha synthesis | arXiv:2308.00016 |
| **Alpha-GPT 2.0** | Successor with a fuller human-in-the-loop quant workflow (idea → factor → backtest feedback). | Iterative HITL alpha refinement | arXiv:2402.09746 |
| **QuantAgent** | Self-improving alpha generation via inner loop (writer agent drafts code, judge agent critiques) + outer loop (real-market test feeds the judge). | Writer/judge dual-loop self-improvement | arXiv:2402.03755 |
| **AlphaAgent** (KDD 2025) | **Decay-resistant** alpha mining: Idea→Factor→Eval agents with AST-based originality, hypothesis-factor alignment, complexity control. Reports +11.0%/+8.74% annual excess on CSI 500 / S&P 500 (2021-2024, after costs). | 3-agent loop + regularized exploration vs. alpha decay | https://github.com/RndmVariableQ/AlphaAgent · arXiv:2502.16789 |
| **Chain-of-Alpha** | Chain-of-thought style LLM pipeline for formulaic alpha mining. | CoT factor generation + refinement | arXiv:2508.06312 |
| **QuantaAlpha** | Evolutionary LLM alpha mining: treats each mining run as a trajectory; trajectory-level mutation/crossover. | Evolutionary multi-agent factor search | arXiv:2602.07085 |
| **QuantEvolve** | Multi-agent **evolutionary** discovery of full quant strategies. | Evolutionary strategy search | arXiv:2510.18569 |

> Honest read: alpha-mining-with-LLMs is the area most exposed to overfitting and survivorship/look-ahead bias. The newer papers (AlphaAgent, QuantaAlpha) explicitly target *alpha decay* precisely because the first wave's signals didn't hold up out of sample.

---

## 4. Finance LLMs & financial-reasoning models

The model layer the agents run on. Some are general finance LLMs; the 2025+ wave is **reasoning**-focused (R1-style RL).

| Model | Base / size | Notes | Link |
|---|---|---|---|
| **BloombergGPT** | From-scratch, 50.6B (BLOOM-style) | **Proprietary**, never released. 363B-token Bloomberg corpus + 345B general. The reference point everyone benchmarks against. | arXiv:2303.17564 · https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/ |
| **FinGPT** | Open, LoRA on Llama/others | Data-centric open alternative to BloombergGPT (see §1). | https://github.com/AI4Finance-Foundation/FinGPT |
| **PIXIU / FinMA** (The-FinAI / ChanceFocus) | Llama-7B/30B fine-tune + 136K instructions + FLARE benchmark | First open financial LLM + instruction data + holistic eval benchmark. NeurIPS 2023. | https://github.com/The-FinAI/PIXIU · arXiv:2306.05443 · https://huggingface.co/ChanceFocus/finma-7b-nlp |
| **InvestLM** (AbaciNLP) | LoRA on LLaMA-65B | Tuned on CFA questions, SEC filings, quant StackExchange; rated comparable to GPT-3.5/4/Claude-2 by hedge-fund/analyst evaluators. | https://github.com/AbaciNLP/InvestLM · arXiv:2309.13064 |
| **FinTral** | Mistral-7B, multimodal | GPT-4-level claims; text+numeric+table+image; FinTral-DPO-T&R variant beats GPT-3.5 across tasks, GPT-4 on 5/9. | arXiv:2402.10986 |
| **Fin-R1** (SUFE-AIFLM-Lab) | Qwen-based, 7B, RL **reasoning** | Distilled from DeepSeek-R1 → SFT+RL on Fin-R1-Data (60,091 CoT samples). SOTA on FinQA/ConvFinQA at 7B. | https://github.com/SUFE-AIFLM-Lab/Fin-R1 · arXiv:2503.16252 |
| **Agentar-Fin-R1** (Ant Group) | Qwen3, 8B/32B, reasoning + trust | Trustworthiness-focused: knowledge engineering, multi-agent data synthesis, validation governance. | arXiv:2507.16802 |
| **Palmyra-Fin** (Writer) | 70B, 32K ctx | **Proprietary/commercial.** First model reported to pass a CFA Level III sample MCQ (~73%). Long-context retrieval focus. | https://writer.com/llms/palmyra-fin/ · https://writer.com/blog/palmyra-med-fin-models/ |
| **Open-FinLLMs** | Open multimodal finance LLMs | Open multimodal family for financial applications. | arXiv:2408.11878 |

> Finance-tuned **Llama/Qwen** are the default open backbones now (FinGPT, Fin-R1, Agentar-Fin-R1 all build on Llama or Qwen). The 2025-2026 shift is from "knows finance facts" → "**reasons** through finance" via R1-style RL.

---

## 5. LLMs for forecasting & sentiment-driven signals

The "can a language model predict returns?" literature — the empirical-finance counterpart to the engineering work above.

| Work | Finding (as stated by authors) | Link |
|---|---|---|
| **Can ChatGPT Forecast Stock Price Movements?** (Lopez-Lira & Tang, UF) | GPT scores from news headlines predict next-day returns; "financial reasoning is an emergent capacity"; beats traditional sentiment. Forthcoming *Journal of Financial Economics*. The seminal empirical paper. | arXiv:2304.07619 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788 |
| **FinGPT sentiment → stock movement** (dissemination-aware, context-enriched) | Improves sentiment-based movement prediction with dissemination + context signals. | arXiv:2412.10823 |
| **FinSphere** | Real-time stock-analysis **agent** (instruction-tuned LLM + live data + quant tools); introduces AnalyScore eval + Stocksis dataset. Springer FITEE 2025. | arXiv:2501.12399 |

---

## 6. Frameworks & infrastructure that *enable* finance agents

You rarely build an agent from scratch. These are the orchestration + data layers.

| Tool | Role for finance agents | Link |
|---|---|---|
| **OpenBB Platform / Workspace** | Open financial **data platform** explicitly built "for analysts, quants and **AI agents**"; MCP servers, custom-agent repo, OpenBB Copilot, "bring your own copilot." The most agent-ready open data layer. | https://github.com/OpenBB-finance/OpenBB · https://github.com/OpenBB-finance/agents-for-openbb · https://docs.openbb.co/workspace/developers/agents-integration |
| **Microsoft AutoGen** | General multi-agent orchestration framework used by several finance projects. Now in maintenance; successor is **Microsoft Agent Framework**. | https://github.com/microsoft/autogen · https://github.com/microsoft/agent-framework |
| **LangChain** | De-facto LLM app/agent framework; tools, RAG, agent loops widely used in finance demos (e.g., LLM-ABM stock sims). | https://github.com/langchain-ai/langchain |
| **LlamaIndex** | Data/RAG framework for grounding agents in filings, reports, financial docs. | https://github.com/run-llama/llama_index |
| **FinNLP** (AI4Finance) | Open financial NLP data pipelines feeding FinGPT/agents. | https://github.com/AI4Finance-Foundation/FinNLP |
| **FinWorld** | All-in-one open platform for **end-to-end** financial AI research + deployment (data → model → agent → backtest). | arXiv:2508.02292 |

---

## 7. Benchmarks, leaderboards & curated lists

Because vendor claims need independent checks.

| Resource | What it gives you | Link |
|---|---|---|
| **Open FinLLM Leaderboard** (The-FinAI + Linux Foundation + HF) | Standardized, open evaluation of finance LLMs across reporting, sentiment, prediction, multimodal. arXiv:2501.10963. | https://github.com/The-FinAI/Open-Financial-LLMs-Leaderboard · https://huggingface.co/blog/leaderboard-finbench |
| **FinLLM-Leaderboard** (Open-Finance-Lab) | Companion leaderboard / FinRL Contest infrastructure. | https://github.com/Open-Finance-Lab/FinLLM-Leaderboard · https://open-finance-lab.github.io/FinRL_Contest_2025/ |
| **PIXIU / FLARE** | Holistic finance LLM benchmark (5 tasks, 9 datasets) — see §4. | https://github.com/The-FinAI/PIXIU |
| **Awesome-LLM-Quantitative-Trading-Papers** | Curated, categorized list of LLM-for-trading papers (agent design, benchmarks, post-training, forecasting, factor discovery). Includes CryptoTrade (EMNLP 2024), ContestTrade, QuantAgent, etc. | https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers |
| **Awesome-FinLLMs** (DataArcTech) | Curated FinLLMs list (papers, models, datasets, code; EN+ZH bilingual focus). | https://github.com/DataArcTech/Awesome-FinLLMs |
| **FinLLMs survey + repo** (adlnlp) | Survey "A Survey of LLMs in Finance (FinLLMs)" + companion datasets/benchmarks. *Neural Computing & Applications* 2025. | https://github.com/adlnlp/FinLLMs · arXiv:2402.02315 |
| **LLM_X_papers** (czyssrs) | Continually updated LLM-in-Finance/Healthcare/Law reading list. | https://github.com/czyssrs/LLM_X_papers |

---

## 8. Reality check — hype vs. evidence (read this before trusting any number)

The single most important section. The field has a documented credibility problem with reported returns.

| Critique paper | Core warning | Link |
|---|---|---|
| **Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents** | "Dazzling back-tested returns evaporate once the model's knowledge window ends." Cause: **pre-training contamination** — the LLM memorized past prices + post-hoc explanations. Proposes **FactFin** (counterfactual perturbations, RAG, MCTS) to force *causal* learning. Empirically isolates leakage using matched 2021 vs. 2024 windows. | arXiv:2510.07920 |
| **The Alpha Illusion** | Argues reported alpha from LLM trading agents should **not** be treated as deployment evidence. | arXiv:2605.16895 |
| **Leakage-Aware Benchmarking of LLM Forecasting** | Use real-time **nowcasts** as the decision-time input to avoid look-ahead leakage when ranking factors. | arXiv:2606.22719 |

Practical implications for AIForge readers:
- **Out-of-sample after the model's knowledge cutoff is the only honest test.** Pre-cutoff backtests of any LLM agent are suspect by default.
- **Transaction costs, slippage, capacity, and survivorship** are often understated or absent in agent papers.
- **Reproducibility varies wildly** — prefer projects with public code + fixed seeds + documented data windows (TradingAgents, AlphaAgent, StockAgent, Fin-R1 publish code; many headline-number papers do not).
- **"Beats GPT-4 / passes the CFA" ≠ "makes money."** Benchmark wins (FinTral, Palmyra-Fin, Fin-R1) measure knowledge/reasoning, not live P&L.
- Commercial models (**BloombergGPT, Palmyra-Fin**) are closed — you cannot audit or reproduce their claims.

---

## 9. Quick-start map (what to actually clone first)

| If you want to… | Start with |
|---|---|
| A complete multi-agent trading sandbox | **TradingAgents** (arXiv:2412.20138) |
| An open finance LLM you can fine-tune | **FinGPT** + **Fin-R1** |
| RL trading environments to plug agents into | **FinRL** / **FinRL-Meta** |
| Equity-research report automation | **FinRobot** (AI4Finance) |
| Data layer that speaks "agent/MCP" | **OpenBB** |
| LLM-driven factor/alpha generation | **AlphaAgent**, **Alpha-GPT** |
| To *stress-test* any of the above honestly | **Profit Mirage / FactFin** methodology |

---

**Sources:** https://github.com/AI4Finance-Foundation · https://ai4finance.org/ · https://github.com/AI4Finance-Foundation/FinGPT · https://arxiv.org/abs/2306.06031 · https://huggingface.co/FinGPT · https://github.com/AI4Finance-Foundation/FinRL · https://github.com/AI4Finance-Foundation/FinRL-Meta · https://github.com/AI4Finance-Foundation/FinRobot · https://github.com/AI4Finance-Foundation/FinNLP · https://github.com/TauricResearch/TradingAgents · https://arxiv.org/abs/2412.20138 · https://arxiv.org/abs/2402.18485 · https://arxiv.org/abs/2311.13743 · https://arxiv.org/abs/2309.03736 · https://arxiv.org/abs/2407.06567 · https://arxiv.org/abs/2502.13165 · https://hedgeagents.github.io · https://github.com/MingyuJ666/Stockagent · https://arxiv.org/abs/2407.18957 · https://arxiv.org/abs/2510.04787 · https://arxiv.org/abs/2507.09255 · https://arxiv.org/abs/2308.00016 · https://arxiv.org/abs/2402.09746 · https://arxiv.org/abs/2402.03755 · https://github.com/RndmVariableQ/AlphaAgent · https://arxiv.org/abs/2502.16789 · https://arxiv.org/abs/2508.06312 · https://arxiv.org/abs/2602.07085 · https://arxiv.org/abs/2510.18569 · https://arxiv.org/abs/2303.17564 · https://github.com/The-FinAI/PIXIU · https://arxiv.org/abs/2306.05443 · https://github.com/AbaciNLP/InvestLM · https://arxiv.org/abs/2309.13064 · https://arxiv.org/abs/2402.10986 · https://github.com/SUFE-AIFLM-Lab/Fin-R1 · https://arxiv.org/abs/2503.16252 · https://arxiv.org/abs/2507.16802 · https://writer.com/llms/palmyra-fin/ · https://arxiv.org/abs/2408.11878 · https://arxiv.org/abs/2304.07619 · https://arxiv.org/abs/2412.10823 · https://arxiv.org/abs/2501.12399 · https://github.com/OpenBB-finance/OpenBB · https://github.com/OpenBB-finance/agents-for-openbb · https://github.com/microsoft/autogen · https://github.com/microsoft/agent-framework · https://arxiv.org/abs/2508.02292 · https://github.com/The-FinAI/Open-Financial-LLMs-Leaderboard · https://arxiv.org/abs/2501.10963 · https://github.com/Open-Finance-Lab/FinLLM-Leaderboard · https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers · https://github.com/DataArcTech/Awesome-FinLLMs · https://github.com/adlnlp/FinLLMs · https://arxiv.org/abs/2402.02315 · https://arxiv.org/abs/2510.07920

**Keywords:** LLM trading agents, agentic AI finance, multi-agent trading framework, TradingAgents, FinGPT, FinRobot, FinRL, FinMem, FinCon, FinAgent, StockAgent, QuantAgent, AlphaAgent, Alpha-GPT, alpha mining LLM, financial reasoning LLM, Fin-R1, BloombergGPT, PIXIU FinMA, InvestLM, FinTral, Palmyra-Fin, OpenBB agents, AI4Finance, information leakage, alpha decay, stock return prediction, financial factor discovery; agentes de negociação com IA, agentes autônomos de investimento, modelos de linguagem para finanças, mineração de fatores (alfa), aprendizado por reforço financeiro, previsão de retornos de ações, B3, BDRs, ETFs.
