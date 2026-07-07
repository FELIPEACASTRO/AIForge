# Repository Scale Benchmark - 2026-07-07

This benchmark records evidence for AIForge's scale. It is intentionally conservative: "largest" is only defensible after a defined metric, sampled competitors, and reproducible counts.

## AIForge Local Metrics

Measured from `C:\Users\davis\Workspace\AIForge` on 2026-07-07.

| Metric | Count |
|---|---:|
| Git-tracked files | 1,860 |
| Markdown files | 1,825 |
| Non-git files scanned, excluding `.git` | 1,873 |
| Directories scanned, excluding `.git` | 572 |
| Estimated text lines | 99,416 |
| Estimated text words | 1,146,752 |
| External URL mentions | 20,301 |
| Unique external URLs | 11,854 |
| Top-level pillar with most files | `05_VERTICAL_APPLICATIONS` (1,135 scanned files) |

## GitHub Comparator Snapshot

Measured with GitHub REST API repository metadata and recursive tree endpoints on 2026-07-07. README URL counts are limited to each repository's root README and should not be treated as full-repository link counts. A later unauthenticated rerun after the local additions returned `HTTP Error 403: rate limit exceeded` with no `GITHUB_TOKEN` present, so the comparator rows below remain the last successful remote snapshot rather than a fresh remote refresh.

| Repository | Stars | Files | Markdown files | README unique URLs | Notes |
|---|---:|---:|---:|---:|---|
| `FELIPEACASTRO/AIForge` | 4 | 1,869 | 1,823 | 23 | Remote GitHub snapshot; local clone had 1,860 tracked files before fetch. |
| `josephmisiti/awesome-machine-learning` | 73,215 | 10 | 7 | 1,252 | Very high-authority compact awesome list. |
| `ChristosChristofidis/awesome-deep-learning` | 28,561 | 1 | 1 | 601 | Compact README list. |
| `owainlewis/awesome-artificial-intelligence` | 15,169 | 6 | 3 | 92 | Compact AI resource list. |
| `Hannibal046/Awesome-LLM` | 27,100 | 28 | 17 | 374 | LLM-specific. |
| `steven2358/awesome-generative-ai` | 12,260 | 8 | 4 | 393 | Generative-AI-specific. |
| `Shubhamsaboo/awesome-llm-apps` | 116,642 | 1,768 | 283 | 31 | Large runnable app/code collection. |
| `aishwaryanr/awesome-generative-ai-guide` | 28,144 | 443 | 116 | 137 | Generative AI guide, notebooks, resources. |
| `ethicalml/awesome-production-machine-learning` | 20,700 | 10 | 3 | 1,111 | Production ML resource list. |
| `academic/awesome-datascience` | 29,563 | 12 | 4 | 971 | Data-science resource list. |
| `ml-tooling/best-of-ml-python` | 23,657 | 438 | 217 | 1,747 | Ranked Python ML libraries; weekly-generated. |
| `zhimin-z/awesome-awesome-machine-learning` | 222 | 3 | 1 | 342 | Meta-list of AI/ML lists. |
| `ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code` | 35,224 | 4 | 1 | 103 | Project-code collection. |

## Conservative Claim

As of this run, AIForge is larger than the sampled major AI/ML/LLM resource repositories by Markdown document count and by locally extracted unique external URLs. It is not yet honest to claim "largest in the world" without:

1. A broader competitor set from GitHub search, Hugging Face, GitLab, personal sites, academic catalogs, and awesome-list aggregators.
2. Full-repository link extraction for competitors, not README-only counts.
3. A public reproducible script and checked-in benchmark artifact.
4. A maintained country/source coverage matrix showing that expansion is not merely volume inflation.

## Reproducible Benchmark Tool

Run from the repository root:

```bash
python tools/benchmark_repository_scale.py --root . --format markdown
```

Set `GITHUB_TOKEN` to raise GitHub API rate limits if needed.

## Next Benchmark Tasks

- Pull top GitHub repositories for queries: `awesome artificial intelligence`, `awesome machine learning`, `awesome deep learning`, `awesome llm`, `awesome generative ai`, `machine learning resources`, `AI papers`, and `AI datasets`.
- Count full-tree Markdown files and links for every candidate.
- Split metrics into resource-count, document-count, country coverage, subject coverage, freshness, and authority.
