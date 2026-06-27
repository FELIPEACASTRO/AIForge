# Wikidata Item — AIForge (ready to create)

Creating a Wikidata item links AIForge to the global knowledge graph, which feeds **Google Knowledge Panel, Bing entities, and voice assistants (Siri / Alexa / Google Assistant)** via entity reconciliation.

> ⚠️ **Notability first.** Wikidata requires items to be "notable" (serious external sources, or use in another Wikimedia project, or fulfilling a structural need). A brand-new repo may be deleted as non-notable. **Recommended:** create this **after** AIForge has some traction (GitHub stars, a blog/news mention, or inclusion in well-known awesome-lists). Keep this ready; submit when you have ≥1 independent source to cite.

## How to create
1. Go to https://www.wikidata.org/ → log in → **"Create a new Item"** (https://www.wikidata.org/wiki/Special:NewItem).
2. Paste the label/description/aliases below.
3. Add the statements (type the property name; Wikidata autocompletes the P-id).
4. Add at least one **reference** (P854 reference URL → a news/blog/awesome-list link) to satisfy notability.

## Label / Description / Aliases

| Field | English | Português |
|---|---|---|
| **Label** | AIForge | AIForge |
| **Description** | curated open index of artificial intelligence, machine learning and deep learning resources | índice aberto e curado de recursos de inteligência artificial, aprendizado de máquina e aprendizado profundo |
| **Aliases** | AIForge repository; AIForge index; AIForge awesome list | repositório AIForge; índice AIForge |

## Statements (properties)

| Property | Value |
|---|---|
| **instance of** (P31) | web resource / free software repository (pick `GitHub repository` if offered, else `online database` Q7094076) |
| **main subject** (P921) | artificial intelligence (Q11660); machine learning (Q2539); deep learning (Q197536); large language model (Q115033602) |
| **official website** (P856) | https://felipeacastro.github.io/AIForge/ |
| **source code repository URL** (P1324) | https://github.com/FELIPEACASTRO/AIForge |
| **copyright license** (P275) | MIT License (Q334661) |
| **inception** (P571) | 2026 |
| **maintained by / author** (P126 / P170) | Felipe Castro |
| **language of work** (P407) | English (Q1860); Portuguese (Q5146) |

## Reference to attach (for notability)
- P854 (reference URL): a link where AIForge is independently mentioned — e.g. an accepted awesome-list PR, a Hacker News thread, or a blog post. Add it once you have one.

> Tip: also add the Wikidata Q-id back to the repo `docs/index.html` JSON-LD `sameAs` array once the item exists, closing the entity loop.
