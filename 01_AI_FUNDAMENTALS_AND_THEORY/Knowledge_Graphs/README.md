# Knowledge Graphs

> A knowledge graph (KG) represents facts as a directed labeled graph of `(head, relation, tail)` triples over entities and relations, enabling structured storage, embedding, completion, and symbolic/neural reasoning — and increasingly grounding LLMs via GraphRAG.

## Why it matters

Knowledge graphs give machines a queryable, provenance-bearing model of the world: entities as nodes, typed relations as edges. KG embeddings (TransE, RotatE, ComplEx) turn this discrete structure into continuous vectors for link prediction and completion, while KG-augmented LLMs and GraphRAG inject verifiable, structured context into generation to reduce hallucination and capture multi-hop relationships. KGs underpin search, recommendation, drug discovery, fraud detection, and enterprise question answering. They sit at the intersection of representation learning, retrieval, and neuro-symbolic reasoning — a foundational topic that touches RAG, GNNs, and meta-learning but deserves treatment in its own right.

## Taxonomy

| Sub-area | What it covers | Representative methods |
|---|---|---|
| **Translational embeddings** | Relations as translations/rotations in vector space | TransE, TransH, RotatE |
| **Tensor / bilinear factorization** | Score triples via (complex) bilinear products | RESCAL, DistMult, ComplEx, SimplE, TuckER |
| **Neural / convolutional** | Deep nets over entity-relation embeddings | ConvE, ConvKB, R-GCN, CompGCN |
| **Compositional / parameter-efficient** | Tokenized or anchor-based entity vocabularies | NodePiece |
| **KG foundation models** | Inductive, transferable, zero-shot reasoning | ULTRA |
| **Rule / path / RL reasoning** | Logical rules and multi-hop path traversal | AMIE, MINERVA, Neural LP |
| **KG construction** | Entity/relation extraction, ontology, fusion | OpenIE, DeepDive, ontology engineering |
| **KG + LLM / GraphRAG** | Graphs as retrieval & reasoning substrate for LLMs | GraphRAG, Think-on-Graph, KG-RAG |

## Key models & methods

| Model | Idea | Year | Link |
|---|---|---|---|
| **TransE** | Relation = translation: `h + r ≈ t` | 2013 | https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html |
| **DistMult** | Bilinear diagonal scoring | 2015 | https://arxiv.org/abs/1412.6575 |
| **ComplEx** | Complex embeddings model asymmetric relations | 2016 | https://arxiv.org/abs/1606.06357 |
| **ConvE** | 2D convolution over reshaped embeddings | 2018 | https://arxiv.org/abs/1707.01476 |
| **RotatE** | Relation = rotation in complex space; infers composition | 2019 | https://arxiv.org/abs/1902.10197 |
| **TuckER** | Tucker tensor decomposition of the binary tensor | 2019 | https://arxiv.org/abs/1901.09590 |
| **NodePiece** | Anchor-based, sublinear-parameter entity tokenization | 2022 | https://arxiv.org/abs/2106.12144 |
| **ULTRA** | Foundation model — zero-shot inductive KG reasoning | 2023 | https://arxiv.org/abs/2310.04562 |
| **Think-on-Graph** | LLM agent traverses KG for deep, interpretable reasoning | 2023 | https://arxiv.org/abs/2307.07697 |

## Key tools & frameworks

| Tool | Scope | Link |
|---|---|---|
| **PyKEEN** | 40+ KG embedding models, training & evaluation | https://github.com/pykeen/pykeen |
| **DGL-KE** | Scalable distributed KG embeddings | https://github.com/awslabs/dgl-ke |
| **GraphVite** | High-speed GPU graph & KG embedding | https://github.com/DeepGraphLearning/graphvite |
| **AmpliGraph** | TensorFlow KG embedding library | https://github.com/Accenture/AmpliGraph |
| **Microsoft GraphRAG** | LLM-built KG + community summarization RAG | https://github.com/microsoft/graphrag |
| **Neo4j GraphRAG (Python)** | KG construction + GraphRAG over Neo4j | https://github.com/neo4j/neo4j-graphrag-python |
| **RDFLib** | RDF graphs, SPARQL, linked data in Python | https://github.com/RDFLib/rdflib |
| **ULTRA** | Foundation KG reasoning model & weights | https://github.com/DeepGraphLearning/ULTRA |

## Benchmarks & datasets

| Dataset | Entities | Relations | Notes | Link |
|---|---|---|---|---|
| **FB15k-237** | 14,541 | 237 | Freebase subset; inverse-relation leakage removed | https://www.microsoft.com/en-us/download/details.aspx?id=52312 |
| **WN18RR** | 40,943 | 11 | WordNet; hierarchical lexical relations | https://github.com/TimDettmers/ConvE |
| **YAGO3-10** | ~123k | 37 | Dense, person-centric relations | https://github.com/TimDettmers/ConvE |
| **CoDEx (S/M/L)** | up to ~78k | up to 69 | Wikidata-based, hard negatives, comprehensive | https://github.com/tsafavi/codex |
| **OGB (ogbl-biokg, ogbl-wikikg2)** | large-scale | varied | Standardized open KG link-prediction splits | https://ogb.stanford.edu/docs/linkprop/ |

Standard metrics: **Mean Reciprocal Rank (MRR)**, **Mean Rank (MR)**, and **Hits@K** (K = 1, 3, 10) under the filtered ranking protocol.

## Key papers

| Paper | Venue / Year | Link |
|---|---|---|
| Translating Embeddings for Modeling Multi-relational Data (TransE) | NeurIPS 2013 | https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html |
| Complex Embeddings for Simple Link Prediction (ComplEx) | ICML 2016 | https://arxiv.org/abs/1606.06357 |
| Convolutional 2D Knowledge Graph Embeddings (ConvE) | AAAI 2018 | https://arxiv.org/abs/1707.01476 |
| Knowledge Graph Embedding: A Survey of Approaches and Applications (Wang et al.) | IEEE TKDE 2017 | https://dblp.org/rec/journals/tkde/WangMWG17.html |
| RotatE: Knowledge Graph Embedding by Relational Rotation | ICLR 2019 | https://arxiv.org/abs/1902.10197 |
| A Survey on Knowledge Graphs: Representation, Acquisition and Applications (Ji et al.) | IEEE TNNLS 2021 | https://arxiv.org/abs/2002.00388 |
| NodePiece: Compositional and Parameter-Efficient Representations | ICLR 2022 | https://arxiv.org/abs/2106.12144 |
| Towards Foundation Models for Knowledge Graph Reasoning (ULTRA) | ICLR 2024 | https://arxiv.org/abs/2310.04562 |
| Think-on-Graph: Deep and Responsible Reasoning of LLM with KG | ICLR 2024 | https://arxiv.org/abs/2307.07697 |
| Unifying Large Language Models and Knowledge Graphs: A Roadmap (Pan et al.) | IEEE TKDE 2024 | https://arxiv.org/abs/2306.08302 |
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | MSR 2024 | https://arxiv.org/abs/2404.16130 |
| Graph Retrieval-Augmented Generation: A Survey | 2024 | https://arxiv.org/abs/2408.08921 |

## Cross-references in AIForge

- [Graph Neural Networks](../Graph_Neural_Networks/) — message passing, R-GCN/CompGCN encoders used for KG completion.
- [RAG and Retrieval](../RAG_and_Retrieval/) — GraphRAG and graph-augmented retrieval pipelines.
- [Neuro-Symbolic AI](../Neuro_Symbolic_AI/) — rule learning and symbolic reasoning over KGs.
- [Information Retrieval](../Information_Retrieval/) — entity linking and structured retrieval foundations.

## Sources

- PyKEEN — https://github.com/pykeen/pykeen
- Microsoft Project GraphRAG — https://www.microsoft.com/en-us/research/project/graphrag/
- ULTRA project page (Mila) — https://deepgraphlearning.github.io/project/ultra
- ComplEx (ICML 2016) — https://arxiv.org/abs/1606.06357
- ConvE (AAAI 2018) — https://arxiv.org/abs/1707.01476
- NodePiece (ICLR 2022) — https://arxiv.org/abs/2106.12144
- CoDEx benchmark — https://github.com/tsafavi/codex
- Open Graph Benchmark (link property prediction) — https://ogb.stanford.edu/docs/linkprop/
- Wang et al. KGE survey (IEEE TKDE 2017) — https://dblp.org/rec/journals/tkde/WangMWG17.html

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
