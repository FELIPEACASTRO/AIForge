# 🛰️ AI Coding Tools & Agents — 2026

> The agentic-coding leading edge: the newest IDEs, CLIs, async agents, orchestration layers and open-source coding agents as they stand in autumn 2026.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026

> ⚠️ Fast-moving field; entries reflect public reporting at refresh time. Confirm versions, licenses and benchmarks at the source links before relying on them. Figures and claims below are attributed to their source, not asserted as settled fact.

## Editors, IDEs & CLI agents

| Item | Date | What's new |
|---|---|---|
| [OpenAI Codex CLI 0.162.0](https://github.com/openai/codex/releases/tag/rust-v0.162.0) | 2026-10-08 | Per the primary GitHub release (Oct 8): adds tools to create/list managed Git worktrees from trusted local projects, pinning tasks in the agent Command Center, and streaming Markdown tables into scrollback. Worktree support has been on by default since v0.156.0. |
| [Google Antigravity 2.0 (stable 2.21.1)](https://en.wikipedia.org/wiki/Google_Antigravity) | 2026-10-07 | As reported (Wikipedia summary — confirm with Google): Antigravity (announced at Google I/O May 19, 2026) is a standalone agent-first desktop app with a Go CLI and Python SDK; by Oct 2026 it expanded to VS Code, Visual Studio, JetBrains, Zed and Xcode extensions, with stable release 2.21.1 dated Oct 7. Reported to use Gemini 3.1 Pro plus Claude and GPT-OSS models. |
| [Cursor — Remote Control (iOS)](https://cursor.com/changelog) | 2026-10-06 | Per the official changelog (Oct 6): view and reply to agents running on your own computer from the Cursor iOS app; on by default except Enterprise. Part of Cursor's push to make the IDE a surface for orchestrating agents rather than the primary workspace. |
| [Claude Code 2.1.287 — "Claude Mods" + agent teammates](https://code.claude.com/docs/en/changelog) | 2026-10-01 | Per the official changelog: v2.1.287 (Oct 1) adds Claude Mods (plugins that modify deeper Claude Code behavior); v2.1.289 adds `agent.spawn` teammates with idle/waiting states; v2.1.292 adds an `effort` param on the Agent tool and `plugin install --marketplace`. |
| [GitHub Copilot — "Agent (Preview)" in Visual Studio](https://github.blog/changelog/2026-07-30-github-copilot-in-visual-studio-july-update/) | 2026-07-30 | Per the official GitHub changelog (Jul 30): a new "Agent (Preview)" in Copilot Chat built on the GitHub Copilot SDK (the same SDK as the Copilot CLI) for more first-try success; adds built-in .NET/Azure skills, Review Selection, and org-level custom instructions. |
| [Windsurf → Devin Desktop + Devin Local (Rust rewrite)](https://docs.devin.ai/desktop/changelog) | 2026-06-02 | Per the official Devin changelog: v3.0.12 (Jun 2) rebranded Windsurf to Devin Desktop; Cascade's successor "Devin Local" is a from-scratch Rust rewrite (self-reported ~30% more token-efficient) supporting subagents, and ships Cognition's own SWE models alongside frontier models. |
| [Amazon Kiro — spec-driven agentic IDE](https://kiro.dev/) | 2026-06-01 | AWS's spec-driven IDE (requirements → design → tasks → agent execution), built on Code OSS; successor to Amazon Q Developer (signups closed May 15, 2026). As reported, current best-in-class for spec-first workflows. GA predates the June refresh (reported Nov 2025) with a June 2026 pricing change; the GA date varies across sources and is not confirmed from a primary AWS page. |

## Agent platforms & orchestration

| Item | Date | What's new |
|---|---|---|
| [OpenAI Agents API (public beta)](https://developers.openai.com/api/docs/changelog) | 2026-09-10 | Per the OpenAI API changelog: public beta released Sep 10 with a managed Codex harness, durable sessions, context compaction and recovery in OpenAI-hosted or self-connected sandboxes; computer use (OpenAI-hosted browser) was added to the Agents API on Sep 29. |
| [Google Jules — async coding agent (Gemini 3.1 Pro)](https://jules.google/docs/changelog/) | 2026-03-09 | Google's async coding agent: the Jules API is reported stable with Gemini 3.1 Pro as the default for Pro users (Mar 9), plus a Planning Critic and CLI parallel runs. Core updates predate the June refresh and a post-mid-2026 official release was not confirmed — check the official changelog for current status. |
| [GitHub Agent HQ + mission control](https://github.blog/news-insights/company-news/welcome-home-agents/) | 2025-10-28 | Announced at GitHub Universe 2025: a central command center to assign, steer and track multiple coding agents (from Anthropic, OpenAI, Google, Cognition, xAI) across GitHub, VS Code, CLI and mobile, with enterprise governance. Rolled out through 2026; predates the June refresh. (GitHub Universe 2026 is Oct 28–29 — expect further updates.) |

## Open-source / self-hostable coding agents

| Item | Date | What's new |
|---|---|---|
| [OpenHands Software Agent SDK v1.53.0](https://github.com/OpenHands/software-agent-sdk/releases) | 2026-10-05 | Per the primary GitHub releases: a rapid cadence through Oct 2026 (v1.53.0 and v1.52.0 both Oct 5) on the redesigned SDK that split the old OpenHands monolith into composable packages (Agent, Conversation, LLM, Tool). Architecture documented in arXiv 2511.03690. |
| [Cline v4.1.22](https://github.com/cline/cline/releases) | 2026-09-30 | Reported (Sep 30) by third-party changelog trackers: v4.1.22 makes GPT-6.1 Sol the default model for OpenAI/OpenRouter/GitHub Copilot providers and adds smarter context compaction. Not confirmed against the primary GitHub releases page — verify version/date there. |
| [OpenCode v1.18.31 (terminal TUI agent)](https://github.com/anomalyco/opencode/releases) | 2026-09-09 | Reported (Sep 9) by aggregators: v1.18.31 fixes ACP session-state persistence on load/resume/fork. OpenCode offers a terminal TUI with parallel multi-session agents, 75+ providers and git-independent undo. The exact version/date and the repo move (from `sst/opencode`) should be confirmed on the primary repo. |

## Coding benchmarks — SOTA & saturation

| Item | Date | What's new |
|---|---|---|
| [SWE-bench Verified saturation — Claude Opus 5 tops at 97%](https://www.vals.ai/benchmarks/swebench) | 2026-09-01 | Per the independent Vals.ai board (last run Sep 1, Mini-SWE-agent harness): Claude Opus 5 leads at 97.00%, DeepSeek V4 Pro at 96.40%, GPT-5.6 Sol at 96.20%. The benchmark is now archived as saturated; harder successors (e.g., SWE-bench Pro) separate frontier models much further. |

## Related

- Radar index: [`00_FRONTIER_AI_2026/README.md`](./README.md)
- Models powering these agents: [`Frontier_LLMs_2026.md`](./Frontier_LLMs_2026.md) · [`Open_Source_LLMs_2026.md`](./Open_Source_LLMs_2026.md)
- Reasoning / agentic framing, MCP & long-horizon autonomy: [`Reasoning_and_Agents_2026.md`](./Reasoning_and_Agents_2026.md)
- Stable agent engineering pillar: [`04_MLOPS_AND_PRODUCTION_AI/AI_Agents`](../04_MLOPS_AND_PRODUCTION_AI/AI_Agents/)
- Evaluation & benchmarks (SWE-bench, agent evals): [`01_AI_FUNDAMENTALS_AND_THEORY/AI_Evaluation`](../01_AI_FUNDAMENTALS_AND_THEORY/AI_Evaluation/)

**Sources:** OpenAI (github.com/openai/codex releases, developers.openai.com changelog), Anthropic (code.claude.com changelog), Cursor (cursor.com/changelog), Cognition/Devin (docs.devin.ai), GitHub (github.blog), Google (jules.google, en.wikipedia.org Antigravity summary), Amazon (kiro.dev), OpenHands (github.com/OpenHands releases), Cline (github.com/cline releases), OpenCode (github.com/anomalyco releases), Vals.ai (vals.ai/benchmarks/swebench) — each row links directly to the primary or earliest reporting verified at refresh time; see the per-row links above.

**Keywords:** AI coding tools 2026, agentic coding, coding agents, Claude Code, Claude Mods, OpenAI Codex CLI, OpenAI Agents API, Cursor, Windsurf, Devin, Devin Desktop, Devin Local, GitHub Copilot, GitHub Agent HQ, Google Jules, Google Antigravity, Amazon Kiro, spec-driven development, OpenHands, Cline, OpenCode, open-source coding agents, terminal TUI agent, Git worktrees, SWE-bench Verified, SWE-bench Pro, coding benchmark SOTA, autonomous software engineering, October 2026.
