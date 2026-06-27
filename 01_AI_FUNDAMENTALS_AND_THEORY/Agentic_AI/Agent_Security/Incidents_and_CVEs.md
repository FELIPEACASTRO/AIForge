# Real-World Incidents & CVEs — AI Agent Security

> Concrete, public, **verified** disclosures (2025–2026) of agent/tool/MCP security failures. Defensive reference: study these to understand realistic threat models.

## Disclosed CVEs & critical vulnerabilities

| Item | What it is |
|---|---|
| [EchoLeak: Zero-Click AI Vulnerability in Microsoft 365 Copilot — CVE-2025-32711 (The Hacker News)](https://thehackernews.com/2025/06/zero-click-ai-vulnerability-exposes.html) | Report on CVE-2025-32711 ('EchoLeak', CVSS 9.3), a zero-click indirect prompt-injection flaw found by Aim Labs that exfiltrated M365 Copilot data (SharePoint/OneDrive/Teams/email scope) with no user interaction via the 'LLM Scope Violation' technique. |
| [Critical RCE in Anthropic MCP Inspector — CVE-2025-49596 (Oligo Security)](https://www.oligo.security/blog/critical-rce-vulnerability-in-anthropic-mcp-inspector-cve-2025-49596) | Primary disclosure of CVE-2025-49596 (CVSS 9.4), an RCE in Anthropic's MCP Inspector: unauthenticated proxy + '0.0.0.0-day' browser flaw lets a malicious website CSRF localhost and run arbitrary commands. Fixed in 0.14.1 (token auth + origin checks). |
| [CVE-2025-6514: Critical mcp-remote RCE (JFrog)](https://jfrog.com/blog/2025-6514-critical-mcp-remote-rce-vulnerability/) | Primary JFrog disclosure of CVE-2025-6514 (CVSS 9.6) in the mcp-remote client (versions 0.0.5–0.1.15, 437k+ downloads): a malicious MCP server's OAuth authorization_endpoint triggers OS command injection via the Node 'open' package. Fixed in 0.1.16. |
| [GitHub Copilot: Remote Code Execution via Prompt Injection (CVE-2025-53773)](https://embracethered.com/blog/posts/2025/github-copilot-remote-code-execution-via-prompt-injection/) | Johann Rehberger's disclosure of a wormable RCE in GitHub Copilot / VS Code: prompt injection from a repo file, source comment, or GitHub issue makes Copilot write 'chat.tools.autoApprove': true into .vscode/settings.json (YOLO mode), disabling confirmations and enabling arbitrary shell command execution. Patched by Microsoft Aug 2025. |
| [InversePrompt: Turning Claude Against Itself (CVE-2025-54794 & CVE-2025-54795)](https://cymulate.com/blog/cve-2025-547954-54795-claude-inverseprompt/) | Cymulate research on two high-severity Claude Code CVEs: CVE-2025-54794 (CVSS 7.7, prefix-based CWD path-restriction bypass / sandbox escape) and CVE-2025-54795 (CVSS 8.7, command injection through whitelisted commands lacking input sanitization, executing without the user-confirmation prompt). |
| [EchoLeak: The First Real-World Zero-Click Prompt Injection Exploit in a Production LLM System (CVE-2025-32711)](https://arxiv.org/abs/2509.10540) | Analysis (Reddy, Gujral) of EchoLeak / CVE-2025-32711, a zero-click indirect prompt injection in Microsoft 365 Copilot enabling data exfiltration via a crafted email with no user action, bypassing the injection classifier via markdown/image/Teams-proxy tricks. Canonical real-world indirect-injection-to-exfiltration case relevant to agentic assistants. |
| [Critical RCE in mcp-remote (CVE-2025-6514) - GitHub Advisory GHSA-6xpm-ggf7-wc3p](https://github.com/advisories/GHSA-6xpm-ggf7-wc3p) | Authoritative GitHub Security Advisory record for CVE-2025-6514 (OS command injection in mcp-remote on connection to untrusted MCP servers), providing the canonical affected-version range and remediation (upgrade to 0.1.16). Cross-reference/primary record for the JFrog disclosure above. |

## Reported incidents & exploits

| Item | What it is |
|---|---|
| [Disrupting the first reported AI-orchestrated cyber espionage campaign](https://www.anthropic.com/news/disrupting-AI-espionage) | Anthropic threat report (Nov 13, 2025) on a Chinese state-sponsored group (GTG-1002) that manipulated Claude Code into running ~80-90% of an espionage operation against ~30 orgs with minimal human input — the canonical real-world agentic-misuse / jailbreak-of-an-agent case. |
| [Indirect Prompt Injection of Claude Computer Use](https://www.hiddenlayer.com/research/indirect-prompt-injection-of-claude-computer-use) | HiddenLayer red-team writeup demonstrating indirect prompt injection (the 'confused deputy' problem) against Anthropic Claude Computer Use, escalating to destructive OS commands. |
| [First Malicious MCP Server Found Stealing Emails in Rogue Postmark-MCP Package (The Hacker News)](https://thehackernews.com/2025/09/first-malicious-mcp-server-found.html) | News report on the first real-world malicious MCP server: a trojanized npm 'postmark-mcp' package (v1.0.16, published Sep 17, 2025) that silently BCC'd every outgoing email to phan@giftshop[.]club; ~1,643 downloads before removal. |
| [CamoLeak: Critical GitHub Copilot Vulnerability Leaks Private Source Code (Legit Security)](https://www.legitsecurity.com/blog/camoleak-critical-github-copilot-vulnerability-leaks-private-source-code) | Primary researcher disclosure (Omer Mayraz) of CamoLeak, a CVSS 9.6 GitHub Copilot Chat flaw: invisible PR comments inject prompts; data exfiltrated as 'ASCII art' through pre-signed GitHub Camo proxy URLs, bypassing CSP. GitHub mitigated Aug 14, 2025 by disabling image rendering. |
| [GitHub MCP Exploited: Accessing private repositories via MCP](https://invariantlabs.ai/blog/mcp-github-vulnerability) | Invariant Labs disclosure (May 26, 2025) of a 'toxic agent flow': a malicious GitHub issue in a public repo hijacks an agent using the official GitHub MCP server, coercing it to read private repos and exfiltrate their contents into an attacker-controlled public PR. Architectural (agent-level), not a server code bug. |

## Why these matter

- **Zero-click prompt injection** (e.g., EchoLeak in M365 Copilot) shows agents can be compromised with **no user action**.
- **MCP RCE chains** (mcp-remote, MCP Inspector) show the *tool/protocol layer* is now a primary attack surface.
- **Malicious MCP servers / tool poisoning** demonstrate live supply-chain risk in agent ecosystems.
- **AI-orchestrated intrusions** (Anthropic disclosure) show agents executing most attack steps autonomously.

> Verified via the section's adversarial double-check (sources fetched/confirmed). Always re-check CVE IDs at the official advisory before formal use.
