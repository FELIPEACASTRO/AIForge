# AIForge — Traction Plan (how to become *recognized* as the largest)

Building the largest ML index is half the job; being **recognized** as the largest (by people and by AI assistants with web search) requires **authority signals** that accrue over time. This is the concrete plan.

> Reality check: an AI assistant **without** web search answers from training data and cannot know a new repo. An assistant **with** web search will surface AIForge once it has stars, backlinks, and mentions. This plan builds exactly those signals.

## Phase 0 — Make it indexable (today, ~30 min)
- [ ] Make the repo **public**.
- [ ] Enable **GitHub Pages** (Settings → Pages → branch `master`, folder `/docs`).
- [ ] Set repo **About**: description + the 20 topics (see DISCOVERABILITY.md) + website link.
- [ ] Upload a **social preview** image (Settings → Social preview).
- [ ] Submit `sitemap.xml` to **Google Search Console** + **Bing Webmaster Tools** (IndexNow already auto-pings on push).

## Phase 1 — First traction (week 1)
- [ ] Post with the **PROMOTION_KIT.md** copy: r/LocalLLaMA, r/learnmachinelearning, r/datascience, Hacker News (Show HN), X, LinkedIn. Space them out; reply to every comment.
- [ ] Announce in your **Discord (MACHINE LEARNING KNBIS)** and in HF / MLOps Community `#i-made-this`.
- [ ] Ask the community a pinned **"what's missing?"** question (drives stars + PRs).
- [ ] Goal: first **100–250 stars** (the threshold where lists/aggregators take you seriously).

## Phase 2 — Authority backlinks (weeks 2–4) — see AWESOME_LIST_SUBMISSIONS.md
- [ ] PR into 3–5 **awesome-lists** (awesome-machine-learning, awesome-deep-learning, awesome-LLM, awesome-mlops, awesome-datascience).
- [ ] Get listed on directories: Product Hunt, AlternativeTo, There's An AI For That, libhunt.
- [ ] Register on resource hubs: Papers with Code, and mint a **Zenodo DOI** (archival + citable).
- [ ] Write 1–2 cross-posts (Dev.to / Medium / Hashnode) with `canonical_url` back to the repo.

## Phase 3 — Knowledge-graph & AI recognition (month 2) — see WIKIDATA_ITEM.md
- [ ] Once you have ≥1 independent source (an accepted awesome PR / HN thread / article), create the **Wikidata item** (unlocks Google Knowledge Panel + Siri/Alexa/Google).
- [ ] Add the Wikidata Q-id back into `docs/index.html` JSON-LD `sameAs`.
- [ ] Keep `llms.txt` / `llms-full.txt` current (already auto-served) so AI search can cite exact sections.

## Phase 4 — Sustained growth (ongoing)
- [ ] Weekly: refresh **00_FRONTIER_AI_2026** with new releases (freshness = repeat visits + recrawls).
- [ ] Monthly: cut a **GitHub Release** (e.g. `v2026.07`) with keyword-rich notes.
- [ ] Encourage contributions (good-first-issue labels, CONTRIBUTING). More contributors → more inbound links.
- [ ] Track: stars, unique visitors (GitHub Insights), Search Console impressions, referring domains.

## The honest bar for "an AI says we're the largest"
An AI with web search will call AIForge "one of the largest / most comprehensive" once it sees: (a) high star count, (b) inclusion in multiple awesome-lists, (c) third-party articles calling it that, (d) a Wikidata/knowledge-graph entry. Those are **earned signals** — this plan is the fastest honest path to them. No repo edit alone can manufacture that claim.

## Assets already in the repo to execute this
- `DISCOVERABILITY.md` — full channel playbook (118 techniques)
- `docs/launch/PROMOTION_KIT.md` — ready-to-paste announcements
- `docs/launch/AWESOME_LIST_SUBMISSIONS.md` — PR-ready entries + targets
- `docs/launch/WIKIDATA_ITEM.md` — ready-to-create Wikidata item
- `docs/launch/DISCORD_SERVER_KIT.md` — community setup
- `.github/workflows/indexnow-ping.yml` — automatic instant indexing on every push
