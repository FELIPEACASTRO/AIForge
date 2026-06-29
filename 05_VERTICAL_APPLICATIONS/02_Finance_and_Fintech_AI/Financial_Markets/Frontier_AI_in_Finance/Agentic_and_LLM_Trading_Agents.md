# Agentic & LLM Trading Agents (Frontier 2024-2026)

> A dense, source-verified map of the **2023-2026 frontier** where large language models (LLMs) and multi-agent ("agentic") systems are applied to trading, investing, alpha mining, and equity research — with real GitHub/arXiv/Hugging Face links, an honest read on hype vs. evidence, and Brazil-access notes. Mainstream venues already indexed elsewhere in this repo (US NYSE/Nasdaq, B3, NSE/BSE, SSE/SZSE, generic data APIs, Kaggle/HF, arXiv q-fin) are referenced only where they connect to agents.

This page covers a research line that exploded after ChatGPT (late 2022) and is now (mid-2026) a crowded, fast-moving subfield. Treat it as a **reading list + radar**, not investment advice. A recurring, well-documented caveat runs through the whole area: backtested "alpha" from LLM agents is frequently inflated by **information leakage / pre-training contamination** — see the *Reality check* section. (Portuguese: agentes de negociação com IA / agentes autônomos de investimento.)

> Every arXiv ID, GitHub repo, and URL below was independently re-verified via web search/fetch in June 2026. Items that have since been **withdrawn or relocated** are flagged inline.

---

## 1. The hub: AI4Finance-Foundation

Most open-source momentum in this space orbits one GitHub organization. If you read nothing else, read this org.

| Project | What it is | Link |
|---|---|---|
| **AI4Finance-Foundation** (org) | Open-source financial AI hub: FinGPT, FinRL, FinRobot, FinNLP, datasets, tutorials. Community-driven, very active. | https://github.com/AI4Finance-Foundation · https://ai4finance.org/ |
| **FinGPT** | Open-source financial LLM stack (data-centric, LoRA-tuned). Positioned as the open alternative to BloombergGPT. Paper: arXiv:2306.06031. Models on HF (`huggingface.co/FinGPT`, ~11 models + 14 datasets). | https://github.com/AI4Finance-Foundation/FinGPT · https://huggingface.co/FinGPT · https://arxiv.org/abs/2306.06031 |
| **FinGPT-Forecaster** | LoRA fine-tune of Llama-2-7b-chat on ~1yr DOW30 news+financials; outputs "positive developments / potential concerns" + a next-week direction call. Released Nov 2023. | https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster |
| **FinRL** | First open framework for **deep reinforcement learning** in trading (not LLM, but the agent substrate many LLM systems plug into). Covers DJIA, S&P 500, NASDAQ-100, HSI, SSE 50, CSI 300; portfolio, crypto, HFT. Paper: arXiv:2011.09607. | https://github.com/AI4Finance-Foundation/FinRL |
| **FinRL-Meta** | Dynamic datasets + standardized market environments ("metaverse") for FinRL agents. | https://github.com/AI4Finance-Foundation/FinRL-Meta |
| **FinRobot** | Multi-agent **platform** layered on top of FinGPT: LLMs + RL + quant tools for equity research, forecasting, report generation. Self-described "Open-Source AI Agent Platform for Financial Analysis using LLMs"; `pip install -U finrobot`. | https://github.com/AI4Finance-Foundation/FinRobot |

> Note: AI4Finance projects are research/education-grade. Stars and ambition are high; live-trading robustness is not guaranteed. Read the issues before trusting any backtest.

---

## 2. Multi-agent LLM trading frameworks (the headline systems)

These mimic a trading firm: specialized LLM agents (analysts, researchers, traders, risk) debate and decide. This is the most-cited cluster.

| Project | What it does | LLM-agent pattern | Link (GitHub / arXiv / HF) |
|---|---|---|---|
| **TradingAgents** (Tauric Research; Xiao, Sun, Luo, Wang — UCLA/MIT) | Simulates a trading firm: fundamental/sentiment/news/technical analysts → bull vs. bear researcher **debate** → trader → risk team. Most popular recent repo in the space. | Role-specialized agents + structured debate + risk gate | https://github.com/TauricResearch/TradingAgents · arXiv:2412.20138 (Dec 2024) · https://tradingagents-ai.github.io/ |
| **A Multimodal Foundation Agent for Financial Trading** (a.k.a. *FinAgent*; NTU) | Tool-augmented, multimodal (numeric+text+visual) trading agent with a **dual-level reflection** module + diversified memory. KDD 2024. Reports large profit gains vs. 9 baselines (treat with caution — single-asset backtests). Title is "A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist"; "FinAgent" is the community nickname. | Multimodal market intelligence + dual reflection + memory retrieval | arXiv:2402.18485 |
| **FinMem** (Stevens Institute) | Performance-enhanced single-agent trader with **layered memory** (short/mid/long, decay) + configurable "character"/risk profile. AAAI Spring Symposium 2024. | Profiling + layered memory + decision module | arXiv:2311.13743 |
| **TradingGPT** | Multi-agent system with **layered memory + distinct characters** and an inter-agent **debate** mechanism over shared holdings. Conceptual precursor to FinMem/TradingAgents. | Layered memory + character design + peer debate | arXiv:2309.03736 |
| **FinCon** (Stevens/Columbia et al.) | Manager–analyst hierarchy with an **Actor-Critic** loop and *Conceptual Verbal Reinforcement* — episodic reflection on wins/losses + risk-control self-critique. NeurIPS 2024. | Hierarchical agents + verbal RL + risk-control belief updates | arXiv:2407.06567 |
| **HedgeAgents** (SCUT + ByteDance) | Central fund manager + asset-class "hedging experts" coordinating via three conference types; balanced/hedging-aware. WWW 2025 (oral). Reports 70% annualized / 400% total over 3yr (backtest — verify before believing). | Manager + specialist experts + scheduled "conferences" | arXiv:2502.13165 · https://hedgeagents.github.io |
| **StockAgent** (*When AI Meets Finance*) | Multi-agent simulation of free trading with **no prior market data** (explicitly designed to avoid test-set leakage); studies how external events move agent behavior/prices. ACM TIST. | Agent-based market simulation (init/trade/post-trade/event phases) | https://github.com/MingyuJ666/Stockagent · arXiv:2407.18957 |
| **TiMi (Trade in Minutes!)** | Decouples strategy *design* from minute-level *deployment* (no continuous inference at runtime). Tests on 200+ pairs across stock + crypto. Submitted Oct 2025. | Tool-specialized agents + closed-loop math reflection + compiled bot | arXiv:2510.04787 |
| **StockSim** (Papadakis et al.) | Dual-mode, **order-level** simulator for evaluating multi-agent LLMs in markets (microstructure-aware: latency, slippage, queue dynamics) + a candlestick mode for scale. | Evaluation/simulation harness for agents | https://github.com/harrypapa2002/StockSim · arXiv:2507.09255 |

**Brazil access note:** these agents are research code; to actually trade Brazilian assets you would wire them to B3 data (already covered in this repo) or trade US tickers/ETFs/BDRs. None ships a B3 connector out of the box — treat that as an integration task.

---

## 3. LLMs for alpha mining & factor discovery

A distinct, fast-growing branch: LLMs **generate trading signals / factor formulas / strategy code** rather than directly placing trades.

| Project | What it does | LLM-agent pattern | Link |
|---|---|---|---|
| **Alpha-GPT** (Shanghai/HKUST) | Human-AI **interactive** alpha mining: prompt-engineering pipeline turns a quant's natural-language idea into candidate alphas. EMNLP 2025 System Demo. | Human-in-the-loop prompt → alpha synthesis | arXiv:2308.00016 |
| **Alpha-GPT 2.0** | Successor with a fuller human-in-the-loop quant workflow (Alpha Mining → Modeling → Analysis). | Iterative HITL alpha refinement | arXiv:2402.09746 |
| **QuantAgent** | Self-improving alpha generation via inner loop (drafts/refines signal code from a knowledge base) + outer loop (real-market test feeds insights back). | Dual-loop self-improvement | arXiv:2402.03755 |
| **AlphaAgent** (KDD 2025) | **Decay-resistant** alpha mining: Idea→Factor→Eval agents with AST-based originality, hypothesis-factor alignment, complexity control. Built on RD-Agent + Qlib. Reports significant excess return on CSI 500 / S&P 500 (2021-2024). | 3-agent loop + regularized exploration vs. alpha decay | https://github.com/RndmVariableQ/AlphaAgent · arXiv:2502.16789 |
| **Chain-of-Alpha** | Dual-chain (generation + optimization) LLM pipeline for formulaic alpha mining on A-shares. **⚠ Withdrawn by the author on arXiv** — listed for completeness, do not cite as live. | CoT factor generation + refinement | arXiv:2508.06312 *(withdrawn)* |
| **QuantaAlpha** | Evolutionary LLM alpha mining: treats each mining run as a trajectory; trajectory-level mutation/crossover + semantic-consistency + complexity control. Feb 2026. | Evolutionary multi-agent factor search | https://github.com/QuantaAlpha/QuantaAlpha · arXiv:2602.07085 |
| **QuantEvolve** | Multi-agent **evolutionary** discovery of full quant strategies with quality-diversity optimization. AI4F workshop @ ICAIF 2025 (oral). | Evolutionary strategy search | https://github.com/tarsyang/quantevolve · arXiv:2510.18569 |

> Honest read: alpha-mining-with-LLMs is the area most exposed to overfitting and survivorship/look-ahead bias. The newer papers (AlphaAgent, QuantaAlpha) explicitly target *alpha decay* precisely because the first wave's signals didn't hold up out of sample.

---

## 4. Finance LLMs & financial-reasoning models

The model layer the agents run on. Some are general finance LLMs; the 2025+ wave is **reasoning**-focused (R1-style RL).

| Model | Base / size | Notes | Link |
|---|---|---|---|
| **BloombergGPT** | From-scratch, 50B (BLOOM-style) | **Proprietary**, never released. 363B-token Bloomberg corpus + general data. The reference point everyone benchmarks against. | arXiv:2303.17564 · https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/ |
| **FinGPT** | Open, LoRA on Llama/others | Data-centric open alternative to BloombergGPT (see §1). | https://github.com/AI4Finance-Foundation/FinGPT |
| **PIXIU / FinMA** (The-FinAI / ChanceFocus) | LLaMA-7B/30B fine-tune + 136K instructions + FLARE benchmark | First open financial LLM + instruction data + holistic eval benchmark. NeurIPS 2023. | https://github.com/The-FinAI/PIXIU · arXiv:2306.05443 · https://huggingface.co/ChanceFocus/finma-7b-nlp |
| **InvestLM** (AbaciNLP) | LoRA on LLaMA-65B | Tuned on CFA questions, SEC filings, quant StackExchange; rated comparable to GPT-3.5 by hedge-fund/analyst evaluators. | https://github.com/AbaciNLP/InvestLM · arXiv:2309.13064 |
| **FinTral** | Mistral-7B, multimodal | GPT-4-level claims; text+numeric+table+image; FinTral-DPO-T&R variant beats GPT-3.5 across tasks, GPT-4 on 5/9. | arXiv:2402.10986 |
| **Fin-R1** (SUFE-AIFLM-Lab) | Qwen-based, 7B, RL **reasoning** | Distilled from DeepSeek-R1 → SFT+RL on Fin-R1-Data (60,091 CoT samples). SOTA on FinQA (76.0) / ConvFinQA (85.0) at 7B. | https://github.com/SUFE-AIFLM-Lab/Fin-R1 · arXiv:2503.16252 |
| **Agentar-Fin-R1** (Ant Group) | Qwen3, 8B/32B, reasoning + trust | Trustworthiness-focused: knowledge engineering, multi-agent data synthesis, validation governance. | arXiv:2507.16802 |
| **Palmyra-Fin** (Writer) | 70B, 32K ctx | **Proprietary/commercial.** First model reported to pass a CFA Level III sample MCQ (~73% vs. GPT-4's ~33%). Long-context retrieval focus. | https://writer.com/llms/palmyra-fin/ · https://writer.com/engineering/palmyra-med-fin-models/ |
| **Open-FinLLMs** | Open multimodal finance LLMs (FinLLaMA / FinLLaVA) | Open multimodal family for financial applications; outperforms general LLMs on several financial tasks. | arXiv:2408.11878 |

> Finance-tuned **Llama/Qwen** are the default open backbones now (FinGPT, Fin-R1, Agentar-Fin-R1 all build on Llama or Qwen). The 2025-2026 shift is from "knows finance facts" → "**reasons** through finance" via R1-style RL.

---

## 5. LLMs for forecasting & sentiment-driven signals

The "can a language model predict returns?" literature — the empirical-finance counterpart to the engineering work above.

| Work | Finding (as stated by authors) | Link |
|---|---|---|
| **Can ChatGPT Forecast Stock Price Movements?** (Lopez-Lira & Tang, UF) | GPT scores from news headlines predict next-day returns on **post-cutoff** headlines; "financial reasoning is an emergent capacity"; beats traditional sentiment. The seminal empirical paper. | arXiv:2304.07619 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788 |
| **FinGPT sentiment → stock movement** (dissemination-aware, context-enriched) | Improves sentiment-based movement prediction with dissemination + context signals (~8% accuracy gain). | arXiv:2412.10823 |
| **FinSphere** | Real-time stock-analysis **agent** (instruction-tuned LLM + live data + quant tools); introduces AnalyScore eval + Stocksis dataset. | arXiv:2501.12399 |

---

## 6. Frameworks & infrastructure that *enable* finance agents

You rarely build an agent from scratch. These are the orchestration + data layers.

| Tool | Role for finance agents | Link |
|---|---|---|
| **OpenBB Platform / Workspace** | Open financial **data platform** self-described "for analysts, quants and **AI agents**"; MCP servers, custom-agent repo, OpenBB Copilot, "bring your own copilot." The most agent-ready open data layer. | https://github.com/OpenBB-finance/OpenBB · https://github.com/OpenBB-finance/agents-for-openbb · https://docs.openbb.co/workspace/developers/agents-integration |
| **Microsoft AutoGen** | General multi-agent orchestration framework used by several finance projects. Now in maintenance; successor is **Microsoft Agent Framework** (Python + .NET). | https://github.com/microsoft/autogen · https://github.com/microsoft/agent-framework |
| **LangChain** | De-facto LLM app/agent framework; tools, RAG, agent loops widely used in finance demos (e.g., LLM-ABM stock sims). | https://github.com/langchain-ai/langchain |
| **LlamaIndex** | Data/RAG framework for grounding agents in filings, reports, financial docs. | https://github.com/run-llama/llama_index |
| **FinNLP** (AI4Finance) | Open financial NLP data pipelines feeding FinGPT/agents. | https://github.com/AI4Finance-Foundation/FinNLP |
| **FinWorld** (TradeMaster-NTU) | All-in-one open platform for **end-to-end** financial AI (data → ML/DL/RL/LLM/agent → backtest); 800M+ samples. | https://github.com/TradeMaster-NTU/FinWorld · arXiv:2508.02292 |

---

## 7. Benchmarks, leaderboards & curated lists

Because vendor claims need independent checks.

| Resource | What it gives you | Link |
|---|---|---|
| **Open FinLLM Leaderboard** (The-FinAI → now FINOS / Linux Foundation + HF) | Standardized, open evaluation of finance LLMs across IE, textual analysis, QA, generation, risk, forecasting, decision-making. arXiv:2501.10963. Repo now lives under **finos-labs**; live HF Space hosted by **finosfoundation**. | https://github.com/finos-labs/Open-Financial-LLMs-Leaderboard · https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard · https://huggingface.co/blog/leaderboard-finbench |
| **FinLLM-Leaderboard** (Open-Finance-Lab) | Companion leaderboard / FinRL Contest infrastructure. | https://github.com/Open-Finance-Lab/FinLLM-Leaderboard · https://open-finance-lab.github.io/FinRL_Contest_2025/ |
| **FinBen** (The-FinAI) | Holistic financial benchmark: 36 datasets, 24 tasks, 7 aspects; first to include **stock-trading** + agent/RAG evaluation. NeurIPS 2024 (Datasets & Benchmarks). | arXiv:2402.12659 · https://huggingface.co/datasets/TheFinAI |
| **PIXIU / FLARE** | Holistic finance LLM benchmark (5 NLP tasks + prediction) — see §4. | https://github.com/The-FinAI/PIXIU |
| **Awesome-LLM-Quantitative-Trading-Papers** | Curated, categorized list of LLM-for-trading papers (agent design, benchmarks, post-training, forecasting, factor discovery). | https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers |
| **Awesome-FinLLMs** (DataArcTech) | Curated FinLLMs list (papers, models, datasets, code; EN+ZH bilingual focus). | https://github.com/DataArcTech/Awesome-FinLLMs |
| **FinLLMs survey + repo** (adlnlp) | "A Survey of Large Language Models in Finance (FinLLMs)" + companion datasets/benchmarks. *Neural Computing & Applications* 2025. | https://github.com/adlnlp/FinLLMs · arXiv:2402.02315 |
| **LLM Agent in Financial Trading: A Survey** (Ding et al.) | Dedicated survey of LLM trading-agent architectures, data inputs, backtest performance, and challenges. | arXiv:2408.06361 |
| **LLM_X_papers** (czyssrs) | Continually updated LLM-in-Finance/Healthcare/Law reading list. | https://github.com/czyssrs/LLM_X_papers |

---

## 8. Reality check — hype vs. evidence (read this before trusting any number)

The single most important section. The field has a documented credibility problem with reported returns.

| Critique paper | Core warning | Link |
|---|---|---|
| **Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents** | "Dazzling back-tested returns evaporate once the model's knowledge window ends." Cause: **pre-training contamination** — the LLM memorized past prices + post-hoc explanations. Releases **FinLake-Bench** + proposes **FactFin** (counterfactual perturbations, RAG, MCTS) to force *causal* learning. | arXiv:2510.07920 |
| **The Alpha Illusion** | Reported alpha from end-to-end LLM trading agents (FinCon, FinMem, TradingAgents, FinAgent, QuantAgent, FLAG-Trader…) should **not** be treated as deployment evidence until it survives temporal-integrity, friction, counterfactual, calibration, execution, and disaggregation tests. | arXiv:2605.16895 |
| **Leakage-Aware Benchmarking of LLM Forecasting** | Use real-time **nowcasts** (e.g. Cleveland Fed archived CPI nowcast) as the decision-time input to avoid look-ahead leakage when ranking factors. | arXiv:2606.22719 |

Practical implications for AIForge readers:
- **Out-of-sample after the model's knowledge cutoff is the only honest test.** Pre-cutoff backtests of any LLM agent are suspect by default.
- **Transaction costs, slippage, capacity, and survivorship** are often understated or absent in agent papers.
- **Reproducibility varies wildly** — prefer projects with public code + fixed seeds + documented data windows (TradingAgents, AlphaAgent, StockAgent, StockSim, Fin-R1 publish code; many headline-number papers do not).
- **\"Beats GPT-4 / passes the CFA\" ≠ \"makes money.\"** Benchmark wins (FinTral, Palmyra-Fin, Fin-R1) measure knowledge/reasoning, not live P&L.
- Commercial models (**BloombergGPT, Palmyra-Fin**) are closed — you cannot audit or reproduce their claims.
- Treat **withdrawn** preprints (e.g. Chain-of-Alpha) and unrefereed headline numbers as provisional.

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
| Microstructure-aware agent evaluation | **StockSim** |
| To *stress-test* any of the above honestly | **Profit Mirage / FactFin** + **FinBen** |

---

**Keywords:** LLM trading agents, agentic AI finance, multi-agent trading framework, TradingAgents, FinGPT, FinRobot, FinRL, FinMem, FinCon, FinAgent, StockAgent, StockSim, QuantAgent, AlphaAgent, Alpha-GPT, QuantaAlpha, alpha mining LLM, financial reasoning LLM, Fin-R1, Agentar-Fin-R1, BloombergGPT, PIXIU FinMA, InvestLM, FinTral, Palmyra-Fin, Open-FinLLMs, OpenBB agents, FinBen, Open FinLLM Leaderboard, AI4Finance, information leakage, alpha decay, stock return prediction, financial factor discovery; agentes de negociação com IA, agentes autônomos de investimento, modelos de linguagem para finanças, mineração de fatores (alfa), aprendizado por reforço financeiro, previsão de retornos de ações, B3, BDRs, ETFs.
