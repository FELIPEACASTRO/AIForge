# AI / ML Paper Discovery — Curated Toolkit

The highest-leverage platforms for **finding, reading, tracking, and reproducing** AI/ML research. Curated subset of the full [preprint catalog](./Preprint_Servers_and_arXiv_Alternatives.md), focused on what an ML practitioner actually uses day-to-day.

## Primary Preprint Source

| Platform | Why it matters | Link |
|---|---|---|
| **arXiv** (cs.LG / cs.AI / cs.CL / cs.CV / stat.ML) | The default home of ML research; near-100% of frontier papers land here first. | https://arxiv.org/list/cs.LG/recent |
| **OpenReview** | Open peer review for NeurIPS / ICLR / ICML / etc. — see reviews + rebuttals, not just the paper. | https://openreview.net/ |

## Read / Skim Better

| Tool | What it adds | Link |
|---|---|---|
| **alphaXiv** | Public comment threads on top of arXiv papers; trending feed. | https://alphaxiv.org/ |
| **ar5iv** | arXiv papers rendered as responsive HTML (LaTeX → HTML). | https://ar5iv.labs.arxiv.org/ |
| **Hugging Face Papers** | Daily curated trending papers, linked to models/datasets/demos. | https://huggingface.co/papers |
| **Semantic Scholar** | AI-powered search, TLDRs, citation context, influential-citation ranking. | https://www.semanticscholar.org/ |
| **Connected Papers** | Visual citation graph to explore a topic's neighborhood. | https://www.connectedpapers.com/ |
| **SciRate** | "Scite"-style rating/feed for arXiv (popular in physics/quantum/ML). | https://scirate.com/ |

## Reproducibility & Code

| Tool | What it adds | Link |
|---|---|---|
| **Papers with Code** | Papers ↔ code ↔ SOTA leaderboards by task/dataset. | https://paperswithcode.com/ |
| **Hugging Face Hub** | Models/datasets/Spaces attached to papers; one-click demos. | https://huggingface.co/ |
| **DBLP** | Authoritative CS bibliography (canonical author/venue records). | https://dblp.org/ |

## Large-Scale Metadata / Graph APIs

| Tool | Use case | Link |
|---|---|---|
| **OpenAlex** | Free, open scholarly graph (works, authors, venues, concepts) — great for building literature pipelines/RAG over papers. | https://openalex.org/ |
| **Semantic Scholar API (S2)** | Programmatic search + embeddings (SPECTER2) for paper retrieval. | https://www.semanticscholar.org/product/api |
| **CORE** | World's largest aggregator of open-access full texts (API + dumps). | https://core.ac.uk/ |
| **Crossref** | DOI metadata for almost everything published. | https://search.crossref.org/ |
| **NASA ADS / INSPIRE-HEP** | Physics/astro deep indexes (relevant for ML-for-science). | https://ui.adsabs.harvard.edu/ · https://inspirehep.net/ |

## Domain Preprint Servers Worth Knowing (ML-adjacent)

- **bioRxiv / medRxiv** — ML-for-biology and clinical ML — https://www.biorxiv.org/ · https://www.medrxiv.org/
- **Cryptology ePrint Archive (IACR)** — ML privacy/security, FHE, MPC — https://eprint.iacr.org/
- **ECCC** — theory / complexity relevant to learning theory — https://eccc.weizmann.ac.il/
- **SSRN** — ML in economics/finance/law — https://www.ssrn.com/
- **Zenodo / Figshare / OSF** — datasets, code, and supplementary artifacts with DOIs — https://zenodo.org/

## Practical Workflows

- **Stay current:** Hugging Face Papers daily + arXiv cs.LG RSS + alphaXiv trending.
- **Go deep on a topic:** Connected Papers graph → Semantic Scholar "influential citations" → Papers with Code SOTA table.
- **Build a literature RAG:** OpenAlex or Semantic Scholar API → embed abstracts (SPECTER2 / bge) → vector DB (see `../Storage_and_Databases/Vector_Databases`).
- **Reproduce results:** Papers with Code → official GitHub → Hugging Face weights/Spaces.

## Cross-references in AIForge
- RAG over papers: [`01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval`](../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/)
- Embedding models: [`01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval`](../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/) · HF models: [`../HuggingFace_Hub`](../HuggingFace_Hub/)
- Frontier research radar: [`00_FRONTIER_AI_2026/Research_Breakthroughs_2026.md`](../../00_FRONTIER_AI_2026/Research_Breakthroughs_2026.md)
