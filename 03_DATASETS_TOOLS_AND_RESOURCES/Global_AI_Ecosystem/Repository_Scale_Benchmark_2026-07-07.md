# Repository Scale Benchmark - 2026-07-07

This benchmark records evidence for AIForge's scale. It is intentionally conservative: "largest" is only defensible after a defined metric, sampled competitors, and reproducible counts.

## AIForge Local Metrics

Measured from `C:\Users\davis\Workspace\AIForge` on 2026-07-07.

| Metric | Count |
|---|---:|
| Git-tracked files | 1,941 |
| Markdown files | 1,893 |
| Non-git files scanned, excluding `.git` | 1,941 |
| Directories scanned, excluding `.git` | 580 |
| Estimated text lines | 102,660 |
| Estimated text words | 1,202,632 |
| External URL mentions | 21,327 |
| Unique external URLs | 12,178 |
| Top-level pillar with most files | `05_VERTICAL_APPLICATIONS` (1,144 scanned files) |

## GitHub Comparator Snapshot

Measured with GitHub REST API repository metadata and recursive tree endpoints on 2026-07-07. README URL counts are limited to each repository's root README and should not be treated as full-repository link counts. The latest unauthenticated API rerun succeeded; the remote `FELIPEACASTRO/AIForge` row can lag the local clone until local commits are pushed.

| Repository | Stars | Files | Markdown files | README unique URLs | Notes |
|---|---:|---:|---:|---:|---|
| `FELIPEACASTRO/AIForge` | 4 | 1,893 | 1,845 | 23 | Remote GitHub default-branch snapshot from the latest successful API pass; local branch has 1,941 scanned files including directory-enrichment Batch 03, Batch 02, Batch 01, the broad AI/ML data-model-prompt source atlas, Batch 09, and the merged Omaha expansion, and remote can lag until branch commits are pushed or merged. |
| `josephmisiti/awesome-machine-learning` | 73,224 | 10 | 7 | 1,252 | Very high-authority compact awesome list. |
| `ChristosChristofidis/awesome-deep-learning` | 28,562 | 1 | 1 | 601 | Compact README list. |
| `owainlewis/awesome-artificial-intelligence` | 15,177 | 6 | 3 | 92 | Compact AI resource list. |
| `Hannibal046/Awesome-LLM` | 27,102 | 28 | 17 | 374 | LLM-specific. |
| `steven2358/awesome-generative-ai` | 12,264 | 8 | 4 | 393 | Generative-AI-specific. |
| `Shubhamsaboo/awesome-llm-apps` | 116,695 | 1,768 | 283 | 31 | Large runnable app/code collection. |
| `aishwaryanr/awesome-generative-ai-guide` | 28,157 | 443 | 116 | 137 | Generative AI guide, notebooks, resources. |
| `ethicalml/awesome-production-machine-learning` | 20,703 | 10 | 3 | 1,111 | Production ML resource list. |
| `academic/awesome-datascience` | 29,569 | 12 | 4 | 972 | Data-science resource list. |
| `ml-tooling/best-of-ml-python` | 23,655 | 438 | 217 | 1,747 | Ranked Python ML libraries; weekly-generated. |
| `zhimin-z/awesome-awesome-machine-learning` | 222 | 3 | 1 | 342 | Meta-list of AI/ML lists. |
| `ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code` | 35,235 | 4 | 1 | 103 | Project-code collection. |

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
