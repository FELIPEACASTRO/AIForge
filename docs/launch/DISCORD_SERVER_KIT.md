# Discord Server Starter Kit — MACHINE LEARNING KNBIS

A ready-to-paste setup to turn the new server into an active AI/ML community. Invite: https://discord.gg/RTRdCVcS3

## Suggested channel structure

**📌 INFO**
- `#welcome` — what the server is + invite link
- `#rules`
- `#announcements` — releases, AIForge updates (set to Announcement channel)
- `#introduce-yourself`

**💬 GENERAL**
- `#general`
- `#off-topic`
- `#jobs-and-opportunities`

**🧠 MACHINE LEARNING**
- `#beginners` — questions, learning paths
- `#papers` — paper discussion (link arXiv)
- `#llms` — LLMs, prompting, fine-tuning
- `#computer-vision`
- `#nlp`
- `#mlops-and-deployment`
- `#kaggle-and-competitions`
- `#datasets-and-tools`

**🛠️ PROJECTS**
- `#show-and-tell` — share your projects
- `#aiforge` — repo discussion, suggestions, contributions
- `#collab` — find collaborators

**🔊 VOICE**
- `Study Room` · `Pair Programming` · `Events`

## #welcome message (paste)
```
👋 Welcome to MACHINE LEARNING KNBIS!

A community for everything AI / Machine Learning / Deep Learning — LLMs, papers,
MLOps, Kaggle, and projects. Beginners and experts both welcome.

📚 Powered by AIForge — a curated index of 5,000+ AI/ML resources:
https://github.com/FELIPEACASTRO/AIForge

👉 Start in #introduce-yourself, then jump into #beginners or #papers.
Please read #rules. Have fun and be kind! 🤖
```

## #rules message (paste)
```
1. Be respectful — no harassment, hate, or discrimination.
2. Stay on-topic per channel; keep #general general.
3. No spam, self-promo, or unsolicited DMs/ads. Share projects in #show-and-tell.
4. No piracy, NSFW, or illegal content.
5. Use English or Portuguese.
6. Credit sources; don't post others' work as your own.
7. Follow Discord's ToS & Community Guidelines.
Mods have final say. Breaking rules → warning → kick → ban.
```

## Recommended setup
- **Roles:** `@Beginner`, `@Practitioner`, `@Researcher`, `@Contributor` (self-assign via a reaction-role bot).
- **Bots:** MEE6 / Carl-bot (moderation + reaction roles), a GitHub webhook into `#announcements` for AIForge commits/releases.
- **Verification level:** Medium; enable the **Community** feature (unlocks Announcement channels, Server Discovery eligibility).
- **Server Discovery:** once you pass member thresholds, enable Discord **Server Discovery** so people searching "machine learning" on Discord find it.
- **Vanity URL:** at Level-3 boost you can claim `discord.gg/your-name` (cleaner link).

## GitHub → Discord announcements (free)
In `#announcements` channel settings → Integrations → Webhooks → create one → copy URL → in the GitHub repo: Settings → Webhooks → add `<webhook-url>/github`, content-type `application/json`, events: pushes + releases. New AIForge releases will auto-post.
